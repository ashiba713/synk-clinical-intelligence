"""Clinical text preprocessing.

Includes whitespace cleaning, optional lightweight de-identification, and a
transparent rule-based negation detector used to annotate evidence polarity.
The negation logic is intentionally simple and fully documented - it is a
heuristic scope rule, not a state-of-the-art NLP system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Cue phrases that put subsequent text (until sentence end or SCOPE_TOKENS
# tokens) under negation scope.
NEGATION_CUES = [
    "no evidence of",
    "no signs of",
    "negative for",
    "ruled out",
    "rule out" ,
    "denies",
    "denied",
    "without evidence of",
    "no new",
    "not",
    "no",
    "resolved",
    "afebrile",
]
SCOPE_TOKENS = 6

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}/\d{2,4}")
_LONG_NUM_RE = re.compile(r"\d{6,}")
_WS_RE = re.compile(r"\s+")


@dataclass
class NegationSpan:
    start: int
    end: int
    cue: str


def clean_text(text: str, deid: bool = False) -> str:
    """Normalise whitespace and optionally redact dates/long digit runs."""
    cleaned = str(text).replace("\r", " ").replace("\n", " ")
    if deid:
        cleaned = _DATE_RE.sub("[DATE]", cleaned)
        cleaned = _LONG_NUM_RE.sub("[NUM]", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def negation_spans(text: str) -> list[NegationSpan]:
    """Return character spans of text covered by a negation cue.

    The rule: a cue negates up to ``SCOPE_TOKENS`` following tokens, bounded
    by sentence-ending punctuation.  This mirrors the classic NegEx-style
    scope heuristics in a transparent way.
    """
    lowered = text.lower()
    token_pattern = re.compile(r"\S+")
    tokens = list(token_pattern.finditer(lowered))
    spans: list[NegationSpan] = []
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]", lowered)]

    for cue in NEGATION_CUES:
        for match in re.finditer(re.escape(cue), lowered):
            cue_end = match.end()
            scope_end = min(len(text), cue_end + SCOPE_TOKENS * 6)
            for boundary in sentence_ends:
                if cue_end < boundary:
                    scope_end = min(scope_end, boundary)
                    break
            spans.append(NegationSpan(start=cue_end, end=scope_end, cue=cue))
    return spans


def is_negated(text: str, phrase: str) -> bool:
    """True if *phrase* occurs inside a negation scope in *text*.

    Used to avoid treating statements like "no signs of infection" as
    positive evidence of infection.
    """
    if not phrase:
        return False
    lowered = text.lower()
    positions = [m.start() for m in re.finditer(re.escape(phrase.lower()), lowered)]
    if not positions:
        return False
    spans = negation_spans(text)
    for pos in positions:
        for span in spans:
            if span.start <= pos < span.end:
                return True
    return False


def sentence_split(text: str) -> list[str]:
    """Lightweight sentence splitting for evidence snippet extraction."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class TextPreprocessor:
    """Configurable text cleaning pipeline (fit is stateless by design)."""

    def __init__(self, config) -> None:
        self.config = config
        self.deid = bool(getattr(config.preprocessing, "text_deid", True))

    def clean(self, text: str) -> str:
        return clean_text(text, deid=self.deid)

    def clean_frame(self, notes: Optional["object"]) -> "object":  # pandas DataFrame
        """Apply cleaning to a notes frame in place (drops blank/duplicate notes)."""
        if notes is None or len(notes) == 0:
            return notes
        out = notes.copy()
        out["note_text"] = out["note_text"].astype(str).map(self.clean)
        out = out[out["note_text"].str.len() > 0]
        out = out.drop_duplicates(subset=["encounter_id", "timestamp", "note_text"])
        return out.reset_index(drop=True)
