"""
Regex-based PII scan over a DataFrame's text columns -- a real,
reusable capability that was entirely absent from this codebase despite
the manuscript's ethics/data-handling sections describing privacy
safeguards. Ported from an external repository audit (this specific
file was one of the genuinely real ones found there, unlike most of
that repository's metrics layer).

SCOPE: this is a basic regex sweep for common identifiable-information
patterns (email, IPv4, US SSN format, phone numbers). It is a coarse
screening tool, not a certified PII-detection system -- false negatives
are expected for anything not matching these specific patterns (e.g.
non-US ID numbers, names, addresses), and the manuscript should
describe it as a supplementary check on this study's own generated/
cached text (LLM outputs, logs), not a guarantee that no identifiable
information exists anywhere in the pipeline.
"""
from __future__ import annotations

import re

import pandas as pd

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Requires explicit phone-style grouping (3-3-4 digits, optional
    # country code, with a separator between every group) -- NOT "any
    # digit followed by 7+ more digits/spaces/dashes", which matched
    # ordinary decimal numbers everywhere in this project's numeric
    # training logs (e.g. "d_loss": 0.21967713476157172 as a "phone
    # number"). Found and fixed while wiring this module in for the
    # first time -- see scripts/scan_run_artifacts_for_pii.py.
    "phone": re.compile(r"\b(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
}


def scan_dataframe(df: pd.DataFrame) -> dict:
    """
    Counts pattern matches across every text (object-dtype) column.
    Returns a dict of {pattern_name: total_match_count}. A returned
    count of 0 across all patterns is evidence of absence for THESE
    patterns specifically, not a general privacy guarantee -- state
    that limitation if this is cited as a data-safety check.
    """
    hits = {pattern_name: 0 for pattern_name in PII_PATTERNS}
    text_columns = df.select_dtypes(include=["object"]).columns
    for column in text_columns:
        column_as_str = df[column].astype(str)
        for pattern_name, pattern in PII_PATTERNS.items():
            hits[pattern_name] += int(column_as_str.str.count(pattern).sum())
    return hits


def scan_text(text: str) -> dict:
    """Same scan, applied to a single string (e.g. one cached LLM response)
    rather than a DataFrame column."""
    return {name: len(pattern.findall(text)) for name, pattern in PII_PATTERNS.items()}
