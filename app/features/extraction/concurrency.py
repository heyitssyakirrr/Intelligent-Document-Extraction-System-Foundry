from __future__ import annotations

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from FileReaderv2.app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_ocr_queue: asyncio.Queue = asyncio.Queue()
_llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(settings.llm_max_concurrent)
_csv_append_lock: asyncio.Lock = asyncio.Lock()
_pending_tasks: set[asyncio.Task] = set()

# ---------------------------------------------------------------------------
# OCR process pool
#
# PaddleOCR is CPU-bound and doesn't release the GIL enough for a thread
# executor to give real concurrency with the asyncio event loop — it starves
# LLM HTTP calls while OCR runs. A separate process has its own GIL, so OCR
# and LLM calls genuinely overlap. max_workers stays at 1 to preserve the
# existing "one file's OCR at a time" behavior; only raise it after
# confirming the server has RAM headroom for another full copy of the model.
# ---------------------------------------------------------------------------
_mp_context = multiprocessing.get_context("spawn")  # fork() + asyncio/threads can deadlock
_ocr_process_pool: ProcessPoolExecutor | None = None


def _init_ocr_worker_process() -> None:
    """
    Runs once inside each spawned worker process, before it accepts any job.
    A spawned process does NOT inherit the parent's logging config, so
    without this, every log line inside paddle_ocr.py silently disappears.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_ocr_process_pool() -> ProcessPoolExecutor:
    global _ocr_process_pool
    if _ocr_process_pool is None:
        _ocr_process_pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=_mp_context,
            initializer=_init_ocr_worker_process,
        )
        logger.info("OCR process pool started (spawn, 1 worker).")
    return _ocr_process_pool


def reset_broken_ocr_pool() -> None:
    """
    Call this after a BrokenProcessPool error (worker crashed/segfaulted).
    Just drops the reference — the dead pool can't be shut down cleanly,
    so the next get_ocr_process_pool() call spins up a fresh one instead
    of every subsequent file failing forever.
    """
    global _ocr_process_pool
    _ocr_process_pool = None
    logger.warning("OCR process pool was broken; a fresh one will start on next use.")


async def shutdown_ocr_process_pool() -> None:
    global _ocr_process_pool
    if _ocr_process_pool is not None:
        loop = asyncio.get_running_loop()
        # shutdown(wait=True) blocks the calling thread; run it off the
        # event loop so it doesn't freeze the app during shutdown.
        await loop.run_in_executor(None, _ocr_process_pool.shutdown, True)
        _ocr_process_pool = None
        logger.info("OCR process pool shut down cleanly.")


def pending_task_count() -> int:
    return _ocr_queue.qsize() + len(_pending_tasks)


def get_pending_tasks() -> "set[asyncio.Task]":
    return set(_pending_tasks)


def _on_task_done(task: asyncio.Task) -> None:
    _pending_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error(
            "Unhandled exception in background task '%s': %s",
            task.get_name(), task.exception(), exc_info=task.exception(),
        )

def _track_task(task: asyncio.Task) -> None:
    _pending_tasks.add(task)
    task.add_done_callback(_on_task_done)
