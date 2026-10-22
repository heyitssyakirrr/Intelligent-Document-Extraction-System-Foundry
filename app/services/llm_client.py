from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx

from FileReaderv2.app.core.config import get_settings

logger = logging.getLogger(__name__)

_EXPECTED_KEYS = {
    "name",
    "master_account_number",
    "sub_account_number",
    "fi_num",
    "bank_name",
}

# Statuses where the same request can succeed later (timeouts, rate limits,
# gateway/model hiccups). Anything else (400 bad payload, 401/403 helper id,
# 404 path, 413 too large) fails the same way every time.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """A Foundry call that produced no usable fields. `retryable` tells the
    pipeline whether another attempt can help; `retry_after` is the wait in
    seconds the gateway asked for on a 429."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass
class LLMExtractionMetadata:
    http_duration_ms: int | None = None
    status_code: int | None = None
    response_length_chars: int | None = None
    parse_strategy: str | None = None
    json_objects_found: int | None = None
    response_truncated: bool = False
    keys_present: list[str] | None = None
    keys_missing: list[str] | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _strip_trailing_commas(text: str) -> str:
    text = re.sub(r",\s*(\])", r"\1", text)
    text = re.sub(r",\s*(\})", r"\1", text)
    return text


def _extract_last_json_object(text: str) -> str | None:
    depth = 0
    start = None
    last_candidate = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                last_candidate = text[start:i + 1]

    return last_candidate


def _extract_json_objects(text: str) -> list[str]:
    depth = 0
    start = None
    candidates: list[str] = []
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])

    return candidates


def _merge_non_empty_dicts(dicts: list[dict]) -> dict:
    merged: dict = {}
    for item in dicts:
        for key, value in item.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            merged[key] = value
    return merged


def _normalise_keys(d: dict) -> dict:
    normalised: dict = {}
    for key, value in d.items():
        clean_key = re.sub(r"[\s\-]+", "_", key.strip()).lower()
        normalised[clean_key] = value
    return normalised


def _filter_expected_keys(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in _EXPECTED_KEYS}


def _clean_fi_num(result: dict) -> dict:
    value = result.get("fi_num")
    if isinstance(value, str):
        result["fi_num"] = re.sub(r"[^0-9A-Za-z]", "", value)
    return result


def _metadata_for_result(
    result: dict,
    parse_strategy: str,
    json_objects_found: int,
    response_truncated: bool,
) -> LLMExtractionMetadata:
    keys_present = [
        key for key in sorted(_EXPECTED_KEYS)
        if result.get(key) not in (None, "")
    ]
    keys_missing = [
        key for key in sorted(_EXPECTED_KEYS)
        if result.get(key) in (None, "")
    ]
    return LLMExtractionMetadata(
        parse_strategy=parse_strategy,
        json_objects_found=json_objects_found,
        response_truncated=response_truncated,
        keys_present=keys_present,
        keys_missing=keys_missing,
    )


def _flagged(results) -> list[str]:
    """Category names Azure marked in a content_filter_results object, e.g.
    {"jailbreak": {"filtered": true, "detected": true}, "hate": {"filtered": false, "severity": "safe"}}."""
    if not isinstance(results, dict):
        return []
    return [
        name for name, value in results.items()
        if isinstance(value, dict) and (value.get("filtered") or value.get("detected"))
    ]


def _message_text(content) -> str:
    """content is normally a string, but some deployments return a list of
    parts - join their text."""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content.strip() if isinstance(content, str) else ""


def _reply_text(body: dict) -> tuple[str, str | None]:
    """The model's answer text and finish_reason from a chat completions body.

    Foundry answers HTTP 200 with a stock refusal when the content filter
    blocks, and with empty content when reasoning used up max_tokens. Neither
    is the model's answer, so both raise with a readable reason."""
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected AI Foundry response shape: {str(body)[:300]}") from exc

    finish_reason = choice.get("finish_reason")
    if finish_reason == "content_filter":
        flags = [
            flag for item in body.get("prompt_filter_results") or []
            if isinstance(item, dict) for flag in _flagged(item.get("content_filter_results"))
        ] or _flagged(choice.get("content_filter_results"))
        categories = ", ".join(flags) or "category not reported"
        logger.error("AI Foundry content filter blocked the request: %s | body: %s", categories, str(body)[:1000])
        raise LLMError(f"Blocked by the AI Foundry content filter: {categories}.", retryable=False)

    text = _message_text((choice.get("message") or {}).get("content"))
    if not text:
        if finish_reason == "length":
            # The same max_tokens will run out again, so retrying only costs tokens.
            raise LLMError(
                "AI Foundry returned an empty reply: the model hit max_tokens. Raise MAX_TOKENS.",
                retryable=False,
            )
        logger.warning("AI Foundry returned empty content. Full response: %s", str(body)[:1000])
        raise LLMError(f"AI Foundry returned an empty reply (finish_reason={finish_reason}).")
    return text, finish_reason


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Wait the gateway asked for on a 429; Azure sends retry-after-ms as well."""
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = response.headers.get(header)
        if value:
            try:
                return float(value) * scale
            except ValueError:
                continue
    return None


def _parse_llm_text(raw: str) -> tuple[dict, LLMExtractionMetadata]:
    try:
        code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_blocks:
            parsed = json.loads(_strip_trailing_commas(code_blocks[-1]))
            result = _clean_fi_num(_filter_expected_keys(_normalise_keys(parsed)))
            metadata = _metadata_for_result(result, "code_block", len(code_blocks), False)
            return result, metadata

        json_blocks = _extract_json_objects(raw)
        if json_blocks:
            parsed_dicts: list[dict] = []
            for block in json_blocks:
                try:
                    parsed = json.loads(_strip_trailing_commas(block))
                    if isinstance(parsed, dict):
                        parsed_dicts.append(
                            _clean_fi_num(_filter_expected_keys(_normalise_keys(parsed)))
                        )
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse one LLM JSON block: %s", exc)
                    continue

            if parsed_dicts:
                merged = _merge_non_empty_dicts(parsed_dicts)
                metadata = _metadata_for_result(merged, "brace_extraction", len(json_blocks), False)
                return merged, metadata

        repaired = raw
        response_truncated = not repaired.endswith("}")
        if response_truncated:
            repaired = repaired.rstrip() + "\n}"
        last_json = _extract_last_json_object(repaired)
        if last_json:
            parsed = json.loads(_strip_trailing_commas(last_json))
            result = _clean_fi_num(_filter_expected_keys(_normalise_keys(parsed)))
            metadata = _metadata_for_result(result, "truncation_repair", 1, response_truncated)
            return result, metadata

        raise ValueError("No JSON object found in LLM response text.")

    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        # The model may format its answer differently next time, so retry.
        raise LLMError(f"Failed to parse LLM response: {exc}") from exc


class LLMClient:
    """Async client for the AI Foundry chat completions endpoint.

    One pooled httpx client is kept for the app's lifetime so concurrent
    files reuse TLS connections instead of a new handshake per call; call
    aclose() on shutdown."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._http: httpx.AsyncClient | None = None

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        if self.settings.helper_id:
            headers["X-PBAI-Helper-Id"] = self.settings.helper_id
        return headers

    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                headers=self._build_headers(),
                timeout=self.settings.llm_timeout_seconds,
                verify=self.settings.llm_verify_ssl,
                limits=httpx.Limits(max_connections=self.settings.llm_max_concurrent),
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _build_payload(self, instructions: str, pdf_bytes: bytes) -> dict:
        # Instructions go first, as a system message identical for every file,
        # so Foundry can cache that prefix; the PDF follows as a file part.
        # The real filename is not sent: it can carry customer details.
        # No temperature or stop: gpt-6-luna is a reasoning model and Foundry
        # rejects both. The parser below copes with any extra text instead.
        data_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")
        # The gateway converts chat messages to the Responses API: it maps "text"
        # parts itself but rejects the chat-style {"type": "file", "file": {...}},
        # so the PDF goes in the Responses API's own flat input_file shape.
        return {
            "model": self.settings.llm_model_name,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the five fields from the attached document."},
                        {"type": "input_file", "filename": "document.pdf", "file_data": data_url},
                    ],
                },
            ],
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }

    async def extract_fields(
        self,
        instructions: str,
        pdf_bytes: bytes,
        timeout: float | None = None,
    ) -> tuple[dict, LLMExtractionMetadata]:
        url = self.settings.llm_url
        logger.debug(
            "Calling AI Foundry at %s (model=%s, instructions=%d chars, pdf=%d bytes)",
            url, self.settings.llm_model_name, len(instructions), len(pdf_bytes),
        )

        t0 = time.perf_counter()
        try:
            response = await self._client().post(
                url,
                json=self._build_payload(instructions, pdf_bytes),
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.TimeoutException as exc:
            raise LLMError(f"AI Foundry timed out after {time.perf_counter() - t0:.0f}s.") from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Unable to reach AI Foundry at {url}: {type(exc).__name__}: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if response.is_error:
            status = response.status_code
            # Foundry explains 4xx errors in the body (bad parameter, unknown helper id, rate limit).
            logger.error("AI Foundry HTTP %d in %dms, body: %s", status, elapsed_ms, response.text[:1000])
            raise LLMError(
                # one line, so it reads cleanly in failed.csv and the summary box
                f"AI Foundry returned HTTP {status}: {' '.join(response.text.split())[:400]}",
                retryable=status in _RETRYABLE_STATUS or status >= 500,
                status_code=status,
                retry_after=_retry_after_seconds(response) if status == 429 else None,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"AI Foundry returned a non-JSON body: {response.text[:300]}") from exc

        usage = body.get("usage") or {}
        text, finish_reason = _reply_text(body)
        logger.debug(
            "AI Foundry call completed: status=%d, duration=%dms, finish_reason=%s, tokens(prompt=%s, completion=%s)",
            response.status_code, elapsed_ms, finish_reason,
            usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?"),
        )

        result, metadata = _parse_llm_text(text)
        metadata.http_duration_ms = elapsed_ms
        metadata.status_code = response.status_code
        metadata.response_length_chars = len(response.text)
        metadata.finish_reason = finish_reason
        metadata.prompt_tokens = usage.get("prompt_tokens")
        metadata.completion_tokens = usage.get("completion_tokens")
        return result, metadata


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()
