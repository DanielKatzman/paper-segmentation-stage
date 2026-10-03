"""Infer stable physical-piece identities across repeated capture sessions."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _piece_group_id(document_id: str, page_id: str,
                    physical_piece_id: str) -> str:
    value = f"{document_id}\n{page_id}\n{physical_piece_id}"
    return "piece_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def assign_physical_piece_identity(
        records: Iterable[dict[str, Any]]) -> str:
    """Add ``physical_piece_id`` and ``split_group_id`` to capture records.

    Matching fragment names are authoritative. Camera timestamp filenames do
    not match across lighting folders, so complete equal-sized sessions are
    paired by natural capture order. Incomplete repeated sessions are refused
    rather than silently allowing one physical piece to cross dataset splits.

    The records are mutated in place. The returned string documents which
    identity method was required by the collection.
    """
    records = list(records)
    by_page: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["document_id"]), str(record["page_id"]))
        by_page.setdefault(key, []).append(record)

    used_ordinal = False
    for (document_id, page_id), page_records in sorted(by_page.items()):
        by_session: dict[str, list[dict[str, Any]]] = {}
        for record in page_records:
            session = str(record["capture_session_id"])
            by_session.setdefault(session, []).append(record)
        ordered_sessions = sorted(by_session.items())
        for _, session_records in ordered_sessions:
            session_records.sort(key=lambda record: _natural_key(str(
                record.get("image_path") or record.get("capture_id", ""))))

        if len(ordered_sessions) == 1:
            for record in page_records:
                record["physical_piece_id"] = str(record["fragment_id"])
        else:
            fragment_sets = [
                {str(record["fragment_id"]) for record in session_records}
                for _, session_records in ordered_sessions
            ]
            names_match = (
                all(len(names) == len(session_records)
                    for names, (_, session_records) in zip(
                        fragment_sets, ordered_sessions))
                and all(names == fragment_sets[0]
                        for names in fragment_sets[1:])
            )
            if names_match:
                for record in page_records:
                    record["physical_piece_id"] = str(record["fragment_id"])
            else:
                counts = {len(items) for _, items in ordered_sessions}
                if len(counts) != 1:
                    details = {
                        session: len(items) for session, items in ordered_sessions}
                    raise ValueError(
                        "Cannot infer repeated physical pieces for page "
                        f"{(document_id, page_id)}: lighting/session counts "
                        f"differ {details}. Give matching filenames in every "
                        "session or finish the incomplete session first.")
                used_ordinal = True
                for _, session_records in ordered_sessions:
                    for ordinal, record in enumerate(session_records, start=1):
                        record["physical_piece_id"] = (
                            f"physical_piece_{ordinal:03d}")

        for record in page_records:
            record["split_group_id"] = _piece_group_id(
                document_id, page_id, str(record["physical_piece_id"]))

    return "session_ordinal_fallback" if used_ordinal else "fragment_id"
