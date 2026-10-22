import os
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLM Extraction Service"
    app_version: str = "1.0.0"
    debug: bool = False
    server_env: Literal["PROD", "UAT"]

    # ---------------------------------------------------------------------------
    # LLM (AI Foundry, OpenAI-compatible chat completions) configuration
    # ---------------------------------------------------------------------------
    llm_base_url: str = ""
    llm_extract_endpoint: str = "/foundry/chat/completions"
    llm_timeout_seconds: float = 600.0
    llm_max_concurrent: int = 3             # simultaneous Foundry calls; keep within the deployment's rate limit
    llm_max_retries: int = 2                # retries per file before marking error
    llm_retry_base_backoff: float = 2.0     # seconds, doubles each retry
    llm_retry_max_wait: float = 60.0        # cap on a 429 Retry-After wait, seconds
    llm_api_key: str | None = None          # gateway authenticates by helper_id; leave empty unless told otherwise
    llm_model_name: str = "gpt-6-luna"
    llm_verify_ssl: bool = False            # kept off as before; turn on once the gateway's CA is trusted
    helper_id: str = ""
    # gpt-6-luna reasons before answering and that reasoning counts against
    # max_tokens; too low a value returns an empty reply (finish_reason=length).
    max_tokens: int = 6000

    # ---------------------------------------------------------------------------
    # DB2 (ELDS) connection — values come from .env / environment, never code
    # ---------------------------------------------------------------------------
    db2_database: str
    db2_hostname: str
    db2_port: int = 50000
    # Toggle: "password" = UID + decrypted password, "gmsa" = Kerberos ticket of the gMSA account.
    db2_auth_mode: Literal["password", "gmsa"] = "password"
    db2_uid: str = ""
    # Optional: if set, `kinit` is run from this keytab before connecting (leave empty if tickets are managed externally).
    db2_gmsa_keytab: str = ""
    db2_gmsa_principal: str = ""

    # ---------------------------------------------------------------------------
    # Input safety
    # ---------------------------------------------------------------------------
    max_input_characters: int = 50_000

    # ---------------------------------------------------------------------------
    # File upload
    # ---------------------------------------------------------------------------
    allowed_upload_extensions: list[str] = [".pdf"]
    # The whole PDF is sent to Foundry base64-encoded (about 1.33x its size),
    # and every page costs input tokens, so keep both limits tight.
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_pdf_pages: int = 6

    max_files_per_batch: int = 500

    # ---------------------------------------------------------------------------
    # /extract — "Submit & Forget" concurrency cap/rate limiting
    # ---------------------------------------------------------------------------
    extract_max_pending_tasks: int = 350

    # Seconds to wait for in-flight single-extraction tasks to finish during
    # graceful shutdown before giving up and letting the process exit anyway.
    extract_shutdown_drain_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file="FileReaderv2/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ---------------------------------------------------------------------------
    # File retention (results / failed / summary logs)
    # ---------------------------------------------------------------------------
    retention_max_days: int = 30

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("llm_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("helper_id")
    @classmethod
    def _strip_helper_id(cls, v: str) -> str:
        # a stray space or Windows line ending makes the header illegal
        return v.strip()

    @field_validator("llm_timeout_seconds")
    @classmethod
    def _positive_llm_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("llm_timeout_seconds must be positive")
        return v

    @field_validator("max_files_per_batch")
    @classmethod
    def _positive_max_files(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_files_per_batch must be positive")
        return v

    @field_validator("extract_max_pending_tasks")
    @classmethod
    def _positive_extract_max_pending(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("extract_max_pending_tasks must be positive")
        return v
    
    @field_validator("retention_max_days")
    @classmethod
    def _positive_retention_max_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("retention_max_days must be positive")
        return v

    @model_validator(mode="after")
    def _check_db2_auth(self) -> "Settings":
        if self.db2_auth_mode == "password" and not self.db2_uid:
            raise ValueError("db2_uid is required when db2_auth_mode='password'")
        if self.db2_auth_mode == "gmsa" and self.db2_gmsa_keytab and not self.db2_gmsa_principal:
            raise ValueError("db2_gmsa_principal is required when db2_gmsa_keytab is set")
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------
    @property
    def llm_url(self) -> str:
        return f"{self.llm_base_url}{self.llm_extract_endpoint}"


_SERVER_ENVS = ("PROD", "UAT")


@lru_cache
def get_settings() -> Settings:
    # SERVER_ENV comes from the venv's bin/activate (or the systemd unit's Environment=).
    server_env = os.getenv("SERVER_ENV", "").strip().upper()
    if server_env not in _SERVER_ENVS:
        raise RuntimeError(f"SERVER_ENV must be one of {_SERVER_ENVS}, got {server_env}")
    # Later files override earlier ones: shared .env first, then .env.prod / .env.uat.
    return Settings(
        _env_file=("FileReaderv2/.env", f"FileReaderv2/.env.{server_env.lower()}"),
        server_env=server_env,
    )
