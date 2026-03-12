from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from underage_moderation.pipeline import run_underage_moderation


def _is_http_event(event: dict) -> bool:
    return "body" in event or "requestContext" in event


def _json_response(status_code: int, payload: dict[str, object]) -> dict[str, object]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _parse_payload(event: dict) -> dict[str, object]:
    if "body" in event:
        body = event["body"]
        if isinstance(body, str):
            return json.loads(body)
        if isinstance(body, dict):
            return body
        raise ValueError("HTTP request body must be a JSON string or object.")

    if event.get("Records") and "s3" in event["Records"][0]:
        record = event["Records"][0]
        return {
            "s3_bucket": record["s3"]["bucket"]["name"],
            "s3_key": record["s3"]["object"]["key"],
        }

    return event


def _download_to_tempfile(source_url: str) -> Path:
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix or ".bin"
    with urllib.request.urlopen(source_url, timeout=30) as response:
        data = response.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        prefix="underage-url-",
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(data)
        return Path(temporary_file.name)


def _download_s3_object(bucket: str, key: str) -> Path:
    import boto3

    suffix = Path(key).suffix or ".bin"
    with tempfile.NamedTemporaryFile(
        delete=False,
        prefix="underage-s3-",
        suffix=suffix,
    ) as temporary_file:
        boto3.client("s3").download_file(bucket, key, temporary_file.name)
        return Path(temporary_file.name)


def lambda_handler(event, context):
    payload = _parse_payload(event)
    temporary_input_path: Path | None = None

    try:
        if "audio_path" in payload:
            input_path = Path(str(payload["audio_path"])).expanduser().resolve()
        elif "audio_url" in payload:
            temporary_input_path = _download_to_tempfile(str(payload["audio_url"]))
            input_path = temporary_input_path
        elif "s3_bucket" in payload and "s3_key" in payload:
            temporary_input_path = _download_s3_object(
                str(payload["s3_bucket"]),
                str(payload["s3_key"]),
            )
            input_path = temporary_input_path
        else:
            raise ValueError(
                "Provide one of: audio_path, audio_url, or an S3 event payload."
            )

        result = run_underage_moderation(
            audio_path=input_path,
            preferred_device=payload.get("device"),
            max_duration_seconds=payload.get("max_duration_seconds"),
            age_threshold=float(payload.get("age_threshold", 0.30)),
            child_probability_margin=float(payload.get("child_probability_margin", 0.0)),
        )

        if _is_http_event(event):
            return _json_response(200, result)
        return result
    except Exception as exc:
        error_payload = {"error": str(exc)}
        if _is_http_event(event):
            return _json_response(400, error_payload)
        return error_payload
    finally:
        if temporary_input_path and temporary_input_path.exists():
            temporary_input_path.unlink()
