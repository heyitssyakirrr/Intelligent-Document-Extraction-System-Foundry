from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from FileReaderv2.app.core.config import get_settings
from FileReaderv2.app.features.extraction.concurrency import _csv_append_lock
from FileReaderv2.app.features.extraction.retention import (
    enforce_dated_folder_retention,
    enforce_extractions_retention,
)
from FileReaderv2.app.models.schemas import ExtractionResult
from FileReaderv2.app.services.db2_luw_client import (
    append_filereader_row,
    connect_db2,
    connect_db2_gmsa,
    kinit_from_keytab,
)
from FileReaderv2.app.services.password_decryptor import decrypt_password

logger = logging.getLogger(__name__)
settings = get_settings()

_RESULTS_ROOT = Path("FileReaderv2","results")
_FAILED_ROOT = Path("FileReaderv2","failed")

_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
_FAILED_ROOT.mkdir(parents=True, exist_ok=True)

_COLUMN_WIDTHS = {
    "timestamp": 20,
    "filename": 35,
    "bank_name": 60,
    "fi_num": 30,
    "master_account_number": 30,
    "sub_account_number": 30,
}

_SUCCESS_CSV_HEADER = ",".join(
    col.ljust(width) for col, width in _COLUMN_WIDTHS.items()
).rstrip() + "\r\n"

_FAILED_CSV_HEADER = "filename,error_message,timestamp\r\n"


def _pad_field(value: str, width: int) -> str:
    #ljust pads with trailing spaces so columns line up in a monospace editor
    return value.ljust(width)

def _escape_csv_field(value: str | None) -> str:
    if value is None:
        return ""
    s = str(value)
    if "," in s or '"' in s or "\n" in s or "\r" in s:
        s = '"' + s.replace('"', '""') + '"'
    return s


async def append_success_row(result: ExtractionResult, filename: str) -> Path:
    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    today = now.strftime("%Y%m%d")
    csv_path = _RESULTS_ROOT / f"{today}_extractions.csv"

    row = ",".join(
        _pad_field(_escape_csv_field(v), width)
        for v, width in zip(
            [timestamp, filename, result.bank_name, result.fi_num,
             result.master_account_number, result.sub_account_number],
             _COLUMN_WIDTHS.values(),
        )
    ) + "\r\n"

    async with _csv_append_lock:
        is_new_file = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            if is_new_file:
                fh.write(_SUCCESS_CSV_HEADER)
            fh.write(row)
            fh.flush()

        enforce_extractions_retention(_RESULTS_ROOT, settings.retention_max_days)

    logger.info("Success row written to %s for file '%s'", csv_path, filename)

    if settings.db2_auth_mode == "gmsa":
        if settings.db2_gmsa_keytab:
            kinit_from_keytab(settings.db2_gmsa_keytab, settings.db2_gmsa_principal)
        conn = connect_db2_gmsa(
            database=settings.db2_database,
            hostname=settings.db2_hostname,
            port=settings.db2_port,
        )
    else:
        pwd = decrypt_password(settings.db2_uid)
        conn = connect_db2(
            database=settings.db2_database,
            hostname=settings.db2_hostname,
            port=settings.db2_port,
            uid=settings.db2_uid,
            pwd=pwd
        )

    try:
        append_filereader_row(
            conn=conn,
            filename=filename,
            bankname=result.bank_name,
            ficode=result.fi_num,
            masteraccno=result.master_account_number,
            subaccno=result.sub_account_number,
        )
    finally:
        conn.close()

    logger.info("Success row written to ELDS DB2 for file '%s'", filename)

    return csv_path


def _write_failure_sync(
    pdf_bytes: bytes,
    filename: str,
    error_message: str,
    timestamp: str,
    today: str,
) -> tuple[Path, Path]:
    day_folder = _FAILED_ROOT / today
    files_folder = day_folder / "failed_files"
    files_folder.mkdir(parents=True, exist_ok=True)

    dest_pdf = files_folder / Path(filename).name
    dest_pdf.write_bytes(pdf_bytes)

    csv_path = day_folder / "failed.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        if write_header:
            fh.write(_FAILED_CSV_HEADER)
        row = ",".join(_escape_csv_field(v) for v in [
            filename,
            error_message,
            timestamp,
        ]) + "\r\n"
        fh.write(row)
        fh.flush()

    enforce_dated_folder_retention(_FAILED_ROOT, settings.retention_max_days)

    logger.info("Failure recorded for '%s' in %s (error: %s)", filename, day_folder, error_message)
    return dest_pdf, csv_path


async def append_failure_row(
    pdf_bytes: bytes,
    filename: str,
    error_message: str,
) -> tuple[Path, Path]:
    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    today = now.strftime("%Y%m%d")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _write_failure_sync,
        pdf_bytes,
        filename,
        error_message,
        timestamp,
        today,
    )
