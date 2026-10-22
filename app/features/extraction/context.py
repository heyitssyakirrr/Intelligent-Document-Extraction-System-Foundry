from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FileProcessingContext:
    filename: str
    processing_timestamp: str
    received_at: datetime
    file_size_bytes: int
    queue_depth_at_upload: int
    intake_id: str = ""

    llm_status: str = "pending"
    llm_attempts: int = 0
    llm_max_attempts: int = 0
    llm_duration_ms: int | None = None
    llm_http_duration_ms: int | None = None
    llm_status_code: int | None = None
    llm_prompt_length_chars: int | None = None
    llm_response_length_chars: int | None = None
    llm_parse_strategy: str | None = None
    llm_json_objects_found: int | None = None
    llm_response_truncated: bool = False
    llm_keys_present: list[str] = field(default_factory=list)
    llm_keys_missing: list[str] = field(default_factory=list)
    llm_finish_reason: str | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_last_error_type: str | None = None
    llm_last_error_message: str | None = None

    storage_status: str = "pending"
    storage_output_path: str | None = None
    failed_pdf_path: str | None = None
    failed_csv_path: str | None = None

    final_status: str = "pending"
    failed_stage: str | None = None
    completed_at: datetime | None = None