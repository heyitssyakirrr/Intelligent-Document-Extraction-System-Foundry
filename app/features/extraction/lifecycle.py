from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from FileReaderv2.app.core.config import get_settings
from FileReaderv2.app.features.extraction.context import FileProcessingContext

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# On-disk layout (all under one root so ops/backup is a single folder)
# ---------------------------------------------------------------------------
_LIFECYCLE_ROOT = Path("uploads") / "inflight"
_FILES_DIR = _LIFECYCLE_ROOT / "files"
_RECORDS_DIR = _LIFECYCLE_ROOT / "records"

_FILES_DIR.mkdir(parents=True, exist_ok=True)
_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _ensure_files_dir() -> None:
    _FILES_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_records_dir() -> None:
    _RECORDS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename_component(filename: str) -> str:
    name = Path(filename).name  # drop any directory components
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip("._") or "upload.pdf"
    return name[:150]  # keep total path length sane


@dataclass(frozen=True)
class IntakeRecord:
    """Returned to the caller right after a file is durably accepted."""

    intake_id: str
    inflight_path: Path


def _intake_filename(processing_timestamp: str, intake_id: str, filename: str) -> str:
    safe_name = _safe_filename_component(filename)
    return f"{processing_timestamp}__{intake_id}__{safe_name}"


def _record_path(intake_id: str) -> Path:
    return _RECORDS_DIR / f"{intake_id}.json"


def _write_record_sync(intake_id: str, record: dict[str, Any]) -> None:
    # Write-to-temp-then-rename so a hard kill mid-write can never leave a
    # half-written record file behind -- Path.replace is atomic on the
    # same filesystem, so the record file always either fully exists with
    # valid JSON, or doesn't exist at all.
    _ensure_records_dir()
    final_path = _record_path(intake_id)
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(final_path)


async def _write_record(intake_id: str, record: dict[str, Any]) -> None:
    # get the current event loop (the engine that manages async tasks)
    loop = asyncio.get_running_loop() 
    # hands _write_record_sync to a thread pool executor, which runs it in a separate thread
    # await suspends the current coroutine until _write_record_sync completes, allowing other tasks to run in the meantime
    # "run the disk write on a background thread, don't freeze the event loop while waiting, then continue."
    await loop.run_in_executor(None, _write_record_sync, intake_id, record)


def _read_record_sync(intake_id: str) -> dict[str, Any] | None:
    path = _record_path(intake_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _update_record_sync(intake_id: str, patch: dict[str, Any]) -> None:
    existing = _read_record_sync(intake_id)
    if existing is None:
        logger.warning(
            "Skipping in-place update for intake_id=%s: its record is "
            "missing or unreadable (likely already resolved, or a torn "
            "write from a hard kill).", intake_id,
        )
        return
    existing.update(patch)
    _write_record_sync(intake_id, existing)


async def _update_record(intake_id: str, patch: dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _update_record_sync, intake_id, patch)


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------
async def register_intake(filename: str, processing_timestamp: str, pdf_bytes: bytes) -> IntakeRecord:
    intake_id = uuid.uuid4().hex[:12]
    inflight_name = _intake_filename(processing_timestamp, intake_id, filename)
    inflight_path = _FILES_DIR / inflight_name

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_files_dir)
    await loop.run_in_executor(None, inflight_path.write_bytes, pdf_bytes)

    await _write_record(intake_id, {
        "intake_id": intake_id,
        "filename": filename,
        "processing_timestamp": processing_timestamp,
        "inflight_path": str(inflight_path),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    })

    logger.debug("Intake persisted for '%s' (intake_id=%s) at %s", filename, intake_id, inflight_path)
    return IntakeRecord(intake_id=intake_id, inflight_path=inflight_path)


# ---------------------------------------------------------------------------
# OCR-complete checkpoint (enables resume to skip OCR)
# ---------------------------------------------------------------------------
async def mark_ocr_complete(intake_id: str, ocr_output_path: str) -> None:
    if not intake_id:
        return
    try:
        await _update_record(intake_id, {"ocr_output_path": ocr_output_path})
    except Exception:
        logger.warning(
            "Failed to checkpoint OCR completion for intake_id=%s; if the "
            "process dies before the LLM stage finishes, this file will "
            "redo OCR on resume instead of skipping it.", intake_id,
        )


# ---------------------------------------------------------------------------
# terminal state + cleanup
# ---------------------------------------------------------------------------
async def mark_terminal(intake_id: str, filename: str, final_status: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _clear_inflight_sync, intake_id)
    logger.debug(
        "Lifecycle resolved for '%s' (intake_id=%s, final_status=%s)",
        filename, intake_id, final_status,
    )


def _clear_inflight_sync(intake_id: str) -> None:
    _record_path(intake_id).unlink(missing_ok=True)
    for path in _FILES_DIR.glob(f"*__{intake_id}__*"):
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# resume (re-enter the live pipeline instead of failing outright)
# ---------------------------------------------------------------------------
async def resume_interrupted_files(requeue_for_ocr_fn, requeue_for_llm_fn) -> int:
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(None, _read_all_records_sync)
    eligible = [r for r in records if not r.get("resume_attempted")]
    if not eligible:
        return set()

    logger.warning(
        "Resuming %d file(s) left in-flight by a previous run that did "
        "not shut down cleanly (each gets exactly one resume attempt).",
        len(eligible),
    )

    resumed_intake_ids: set[str] = set()
    for record in eligible:
        intake_id = record.get("intake_id", "")
        inflight_path = Path(record.get("inflight_path", ""))
        filename = record.get("filename") or inflight_path.name or "unknown.pdf"
        processing_timestamp = record.get("processing_timestamp", "")
        ocr_output_path_str = record.get("ocr_output_path")

        if not inflight_path.exists():
            continue

        try:
            await _update_record(intake_id, {"resume_attempted": True})
        except Exception:
            logger.exception(
                "Failed to set resume_attempted flag for '%s' "
                "(intake_id=%s); skipping resume for this file this "
                "startup to avoid a possible repeat-resume loop. It will "
                "be retried on the next startup.", filename, intake_id,
            )
            continue

        try:
            pdf_bytes = await loop.run_in_executor(None, inflight_path.read_bytes)
        except OSError:
            logger.exception(
                "Failed to read staged PDF for '%s' (intake_id=%s) during "
                "resume; leaving it for recovery as a failure instead.",
                filename, intake_id,
            )
            continue

        ctx = FileProcessingContext(
            filename=filename,
            processing_timestamp=processing_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
            received_at=datetime.now(),
            file_size_bytes=len(pdf_bytes),
            queue_depth_at_upload=0,
            intake_id=intake_id,
        )

        if ocr_output_path_str:
            ocr_output_path = Path(ocr_output_path_str)
            try:
                ocr_text = await loop.run_in_executor(None, ocr_output_path.read_text, "utf-8")
            except OSError:
                logger.warning(
                    "OCR checkpoint exists for '%s' (intake_id=%s) but its "
                    "txt at %s is unreadable; resuming via OCR instead of "
                    "skipping it.", filename, intake_id, ocr_output_path,
                )
                await requeue_for_ocr_fn(ctx, pdf_bytes)
            else:
                ctx.ocr_output_path = str(ocr_output_path)
                logger.info(
                    "Resuming '%s' (intake_id=%s) directly into the LLM "
                    "stage using its saved OCR output.", filename, intake_id,
                )
                await requeue_for_llm_fn(ctx, ocr_text, pdf_bytes)
        else:
            logger.info(
                "Resuming '%s' (intake_id=%s) via the OCR queue (no saved "
                "OCR output found).", filename, intake_id,
            )
            await requeue_for_ocr_fn(ctx, pdf_bytes)
                                                                                  
        resumed_intake_ids.add(intake_id)

    return resumed_intake_ids


def _read_all_records_sync() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record_path in _RECORDS_DIR.glob("*.json"):
        try:
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            logger.error(
                "Inflight record at %s is missing or unreadable; the corresponding staged file (if any) cannot be"
                "auto-recovered into failed.csv.", record_path,
            )
    return records


async def recover_orphaned_files(
    record_failure_fn,
    exclude_intake_ids: "set[str] | None" = None,
) -> int:
    
    exclude_intake_ids = exclude_intake_ids or set()
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(None, _read_all_records_sync)
    already_resumed_once = [
        r for r in records
        if r.get("resume_attempted") and r.get("intake_id") not in exclude_intake_ids
    ]
    if not already_resumed_once:
        return 0

    logger.warning(
        "Recovering %d file(s) that already had one resume attempt in a "
        "previous run and are still unresolved.", len(already_resumed_once),
    )

    recovered = 0
    for record in already_resumed_once:
        intake_id = record.get("intake_id", "")
        inflight_path = Path(record.get("inflight_path", ""))
        filename = record.get("filename") or inflight_path.name or "unknown.pdf"

        if not inflight_path.exists():
            logger.error(
                "Orphaned record for '%s' (intake_id=%s) has no matching staged file at %s -- cannot recover its bytes; leaving"
                "it out of failed.csv.", filename, intake_id, inflight_path,
            )
            await loop.run_in_executor(None, _clear_inflight_sync, intake_id)
            continue

        try:
            pdf_bytes = await loop.run_in_executor(None, inflight_path.read_bytes)
            await record_failure_fn(
                intake_id,
                filename,
                pdf_bytes,
                "resume_failed",
                "Processing was interrupted by an application shutdown or crash, and a previous automatic resume attempt also did"
                "not complete. This file was not retried again automatically; please re-upload it if needed.",
            )
            recovered += 1
        except Exception:
            logger.exception(
                "Failed to recover orphaned file '%s' (intake_id=%s); will "
                "retry on next startup.", filename, intake_id,
            )

    return recovered


# ---------------------------------------------------------------------------
# shutdown drain
# ---------------------------------------------------------------------------
async def drain_and_finalize(pending_tasks: "set[asyncio.Task]", timeout: float | None = None) -> None:
    effective_timeout = timeout if timeout is not None else settings.extract_shutdown_drain_seconds

    if not pending_tasks:
        return

    logger.info(
        "Draining %d in-flight task(s) (timeout=%.0fs)...",
        len(pending_tasks), effective_timeout,
    )
    started = time.perf_counter()
    done, pending = await asyncio.wait(pending_tasks, timeout=effective_timeout)
    elapsed = time.perf_counter() - started

    if pending:
        logger.warning(
            "%d task(s) did not finish within the %.0fs drain window "
            "(elapsed %.1fs) and will be recovered automatically on next "
            "startup instead.", len(pending), effective_timeout, elapsed,
        )
    else:
        logger.info("All in-flight task(s) finished cleanly in %.1fs.", elapsed)
