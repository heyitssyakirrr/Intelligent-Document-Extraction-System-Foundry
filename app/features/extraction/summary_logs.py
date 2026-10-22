from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from FileReader.app.core.config import get_settings
from FileReader.app.features.extraction.context import FileProcessingContext

settings = get_settings()
logger = logging.getLogger(__name__)
_summary_log_lock = asyncio.Lock()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _logs_root() -> Path:
    return _project_root().parent / "FileReader_Logs"


def _fmt_dt(value: datetime | None) -> str:
    return value.isoformat(timespec="milliseconds") if value else "-"


def _fmt_ms(value: int | None) -> str:
    return f"{value:,} ms" if value is not None else "-"


def _fmt_count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "-"


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def _total_duration_ms(ctx: FileProcessingContext) -> int | None:
    if ctx.completed_at is None:
        return None
    return int((ctx.completed_at - ctx.received_at).total_seconds() * 1000)


def build_file_summary(ctx: FileProcessingContext) -> str:
    # Total box width including the two │ wall characters.
    W = 88
    # Inner width = space available for content between the two walls.
    INNER = W - 2  # 86 chars

    def wall(text: str = "") -> str:
        """Pad `text` to INNER chars and wrap it with │ walls."""
        return f"│{text:<{INNER}}│"

    def section(label: str) -> str:
        """Section divider: ├──[ LABEL ]──...──┤"""
        tag = f"[ {label} ]"
        left = "──"
        right = "─" * (INNER - len(left) - len(tag))
        return f"├{left}{tag}{right}┤"

    completed_label = "Completed" if ctx.final_status == "success" else "Failed At"
    failed_stage = ctx.failed_stage.upper() if ctx.failed_stage else "-"

    title = f"  FILE SUMMARY  │  {ctx.filename}  │  run: {ctx.processing_timestamp}"
    if len(title) > INNER:
        title = title[: INNER - 3] + "..."

    top    = "┌" + "─" * INNER + "┐"
    bottom = "└" + "─" * INNER + "┘"
    divide = "├" + "─" * INNER + "┤"

    rows = [
        top,
        wall(title),
        divide,
        wall(f"  Status         : {ctx.final_status.upper()}"),
    ]

    if ctx.final_status == "failed":
        rows.append(wall(f"  Failed Stage   : {failed_stage}"))

    rows += [
        wall(f"  Received       : {_fmt_dt(ctx.received_at)}"),
        wall(f"  {completed_label:<15}: {_fmt_dt(ctx.completed_at)}"),
        wall(f"  Total Duration : {_fmt_ms(_total_duration_ms(ctx))}"),
        section("UPLOAD"),
        wall(f"  File Size      : {ctx.file_size_bytes:,} bytes"),
        wall(f"  Queue Depth    : {ctx.queue_depth_at_upload}"),
        section("OCR"),
        wall(f"  Status         : {ctx.ocr_status}"),
        wall(f"  Duration       : {_fmt_ms(ctx.ocr_duration_ms)}"),
        wall(f"  Chars Extracted: {ctx.ocr_char_count if ctx.ocr_char_count is not None else '-'}"),
        wall(f"  Output         : {ctx.ocr_output_path or '-'}"),
        wall(f"  Error Type     : {ctx.ocr_error_type or '-'}"),
        wall(f"  Error Message  : {ctx.ocr_error_message or '-'}"),
        section("LLM"),
        wall(f"  Status         : {ctx.llm_status}"),
        wall(f"  Attempts       : {ctx.llm_attempts} / {ctx.llm_max_attempts}"),
        wall(f"  Duration       : {_fmt_ms(ctx.llm_duration_ms)}"),
        wall(f"  HTTP Duration  : {_fmt_ms(ctx.llm_http_duration_ms)}"),
        wall(f"  HTTP Status    : {ctx.llm_status_code if ctx.llm_status_code is not None else '-'}"),
        wall(f"  Prompt Length  : {ctx.llm_prompt_length_chars if ctx.llm_prompt_length_chars is not None else '-'} chars"),
        wall(f"  Response Length: {ctx.llm_response_length_chars if ctx.llm_response_length_chars is not None else '-'} chars"),
        wall(f"  Parse Strategy : {ctx.llm_parse_strategy or '-'}"),
        wall(f"  JSON Objects   : {ctx.llm_json_objects_found if ctx.llm_json_objects_found is not None else '-'}"),
        wall(f"  Truncated      : {ctx.llm_response_truncated}"),
        wall(f"  Finish Reason  : {ctx.llm_finish_reason or '-'}"),
        wall(f"  Tokens         : prompt {_fmt_count(ctx.llm_prompt_tokens)} / completion {_fmt_count(ctx.llm_completion_tokens)}"),
        wall(f"  Fields Present : {_fmt_list(ctx.llm_keys_present)}"),
        wall(f"  Fields Missing : {_fmt_list(ctx.llm_keys_missing)}"),
        wall(f"  Last Error Type: {ctx.llm_last_error_type or '-'}"),
        wall(f"  Last Error     : {ctx.llm_last_error_message or '-'}"),
        section("STORAGE"),
        wall(f"  Status         : {ctx.storage_status}"),
        wall(f"  Output         : {ctx.storage_output_path or '-'}"),
        wall(f"  Failed PDF     : {ctx.failed_pdf_path or '-'}"),
        wall(f"  Failed CSV     : {ctx.failed_csv_path or '-'}"),
        bottom,
        "",
    ]
    return "\n".join(rows)


def enforce_summary_log_retention() -> None:
    logs_root = _logs_root()
    if not logs_root.exists():
        return

    cutoff = datetime.now().date() - timedelta(days=settings.retention_max_days)
    for path in logs_root.glob("*_logs.log"):
        date_part = path.name.split("_", 1)[0]
        try:
            item_date = datetime.strptime(date_part, "%Y%m%d").date()
        except ValueError:
            continue
        if item_date < cutoff:
            path.unlink(missing_ok=True)


async def append_file_summary(ctx: FileProcessingContext) -> Path:
    logs_root = _logs_root()
    today = datetime.now().strftime("%Y%m%d")
    log_path = logs_root / f"{today}_logs.log"

    async with _summary_log_lock:
        try:
            logs_root.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", newline="") as fh:
                fh.write(build_file_summary(ctx))
            enforce_summary_log_retention()
        except OSError as exc:
            logger.error("Failed to write file summary log to %s: %s", log_path, exc)

    return log_path
