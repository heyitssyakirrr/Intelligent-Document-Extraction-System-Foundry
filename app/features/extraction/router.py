from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from FileReaderv2.app.core.config import get_settings
from FileReaderv2.app.features.extraction.concurrency import pending_task_count
from FileReaderv2.app.features.extraction.context import FileProcessingContext
from FileReaderv2.app.features.extraction.lifecycle import register_intake
from FileReaderv2.app.features.extraction.pipeline import submit_for_extraction
from FileReaderv2.app.services.file_service import validate_and_read_upload

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/extract", tags=["Single Extraction"])


@router.post(
    "",
    summary="Submit & Forget — queue one PDF for background LLM extraction",
    response_class=JSONResponse,
    responses={
        200: {
            "description": "File accepted and queued. Connection closes immediately.",
            "content": {"application/json": {"example": {"success": True, "message": "File queued successfully."}}},
        },
        422: {
            "description": "Validation failure (unsupported type, file too large, etc.).",
            "content": {"application/json": {"example": {"success": False, "message": "<reason>"}}},
        },
        503: {
            "description": "Server is at capacity — too many files already queued/processing.",
            "content": {"application/json": {"example": {"success": False, "message": "Server is at capacity. Please retry shortly."}}},
        },
    },
)
async def extract_single(
    file: UploadFile = File(..., description="Single PDF file to process"),
) -> JSONResponse:
    if pending_task_count() >= settings.extract_max_pending_tasks:
        logger.warning(
            "Rejecting upload '%s' — at capacity (%d/%d in-flight tasks)",
            file.filename, pending_task_count(), settings.extract_max_pending_tasks,
        )
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Server is at capacity. Please retry shortly.",
            },
        )

    try:
        pdf_bytes, _ext = await validate_and_read_upload(file)
    except Exception as exc:
        logger.warning("Validation failed for '%s': %s", file.filename, exc)
        return JSONResponse(
            status_code=getattr(exc, "status_code", 422),
            content={"success": False, "message": getattr(exc, "detail", str(exc))},
        )

    filename = file.filename or "uploaded_file.pdf"
    received_at = datetime.now()
    processing_timestamp = received_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # Durably persist the upload before it is queued for processing. From
    # this point on, the file can always be recovered on next startup even
    # if the process stops before any later stage runs.
    intake = await register_intake(filename, processing_timestamp, pdf_bytes)

    ctx = FileProcessingContext(
        filename=filename,
        processing_timestamp=processing_timestamp,
        received_at=received_at,
        file_size_bytes=len(pdf_bytes),
        queue_depth_at_upload=pending_task_count(),
        intake_id=intake.intake_id,
    )

    logger.info(
        "File accepted for background processing: '%s' (run=%s, size=%d bytes)",
        filename,
        processing_timestamp,
        len(pdf_bytes),
    )

    submit_for_extraction(ctx, pdf_bytes)
    logger.info(
        "File queued for LLM extraction: '%s' (run=%s, in-flight now %d)",
        filename,
        processing_timestamp,
        pending_task_count(),
    )

    return JSONResponse(
        status_code=200,
        content={"success": True, "message": "File queued successfully."},
    )
