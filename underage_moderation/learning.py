from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

HEADER_BY_KIND = {
    "learning": (
        "# Learnings\n\n"
        "Corrections, review feedback, and recurring moderation patterns.\n\n"
    ),
    "error": "# Errors\n\nOperational failures, tool issues, and runtime surprises.\n\n",
    "feature-request": (
        "# Feature Requests\n\n"
        "Requested capabilities that are not yet part of the public runtime.\n\n"
    ),
}

FILE_BY_KIND = {
    "learning": "LEARNINGS.md",
    "error": "ERRORS.md",
    "feature-request": "FEATURE_REQUESTS.md",
}

ID_PREFIX_BY_KIND = {
    "learning": "LRN",
    "error": "ERR",
    "feature-request": "FEAT",
}

ENTRY_PATTERN = re.compile(r"^## \[(?P<entry_id>[A-Z]+-\d{8}-(?P<index>\d{3}))\]")


def _resolve_learning_file(output_dir: str | Path, kind: str) -> Path:
    if kind not in FILE_BY_KIND:
        raise ValueError(f"Unsupported learning entry kind: {kind}")

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target_file = directory / FILE_BY_KIND[kind]

    if not target_file.exists():
        target_file.write_text(HEADER_BY_KIND[kind], encoding="utf-8")

    return target_file


def _next_entry_id(existing_text: str, prefix: str, date_token: str) -> str:
    highest_index = 0
    for line in existing_text.splitlines():
        match = ENTRY_PATTERN.match(line)
        if not match:
            continue
        entry_id = match.group("entry_id")
        if not entry_id.startswith(f"{prefix}-{date_token}-"):
            continue
        highest_index = max(highest_index, int(match.group("index")))
    return f"{prefix}-{date_token}-{highest_index + 1:03d}"


def append_learning_entry(
    *,
    kind: str,
    summary: str,
    details: str,
    suggested_action: str,
    output_dir: str | Path = ".learnings",
    category: str = "insight",
    priority: str = "medium",
    area: str = "policy",
    source: str = "manual",
    tags: list[str] | None = None,
) -> Path:
    target_file = _resolve_learning_file(output_dir, kind)
    existing_text = target_file.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)
    date_token = now.strftime("%Y%m%d")
    timestamp = now.isoformat().replace("+00:00", "Z")
    entry_id = _next_entry_id(existing_text, ID_PREFIX_BY_KIND[kind], date_token)
    normalized_tags = ", ".join(tag.strip() for tag in (tags or []) if tag.strip())
    normalized_tags = normalized_tags or "none"

    title_suffix = category if kind == "learning" else kind
    entry = (
        f"## [{entry_id}] {title_suffix}\n\n"
        f"**Logged**: {timestamp}\n"
        f"**Priority**: {priority}\n"
        f"**Status**: pending\n"
        f"**Area**: {area}\n\n"
        f"### Summary\n{summary.strip()}\n\n"
        f"### Details\n{details.strip()}\n\n"
        f"### Suggested Action\n{suggested_action.strip()}\n\n"
        "### Metadata\n"
        f"- Source: {source}\n"
        f"- Tags: {normalized_tags}\n\n"
        "---\n\n"
    )
    target_file.write_text(existing_text + entry, encoding="utf-8")
    return target_file
