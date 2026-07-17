"""Isolated feature-only diagnostics built on the frozen MM-001 evidence."""

from .method import (
    CODEC_VARIANTS,
    WORLD_VARIANTS,
    assert_parent_parity,
    config_record,
    execute,
    matched_horizon_table,
    report_text,
    summarize,
    validate_evidence,
)

__all__ = [
    "CODEC_VARIANTS",
    "WORLD_VARIANTS",
    "assert_parent_parity",
    "config_record",
    "execute",
    "matched_horizon_table",
    "report_text",
    "summarize",
    "validate_evidence",
]
