"""Result validation/repair: cleanup_result and helper functions."""

from __future__ import annotations

import re

from minutes.models import ExtractionResult

_BAD_OWNER_RE = re.compile(
    r"\b(team|lead|committee|panel|board|management|department|division|"
    r"manager|developer|engineer|architect|analyst|reviewer|group)\b",
    re.IGNORECASE,
)

_VALID_OWNER_RE = re.compile(r"^$|^user$|^assistant$|^[A-Z][a-z]+(\s[A-Z][a-z]+)*$")

_FILLER_PATTERNS = [
    re.compile(r"^no (particular|specific|explicit|clear|stated|given|documented)\b", re.I),
    re.compile(r"^not (specified|mentioned|stated|discussed|provided|given|documented)\b", re.I),
    re.compile(r"^none (provided|given|stated|mentioned|specified)\b", re.I),
    re.compile(r"^straightforward\b", re.I),
    re.compile(r"^n/?a$", re.I),
    re.compile(r"^(no|none|n/?a|tbd|unknown|unspecified)$", re.I),
    re.compile(r"^implicit\b", re.I),
    re.compile(r"^(just|simply)\s+(a\s+)?(decision|choice|standard)\b", re.I),
    re.compile(r"no debate", re.I),
    re.compile(r"no (particular |specific )?reason(ing)?\b", re.I),
    re.compile(r"^it'?s (just )?(what|how) we", re.I),
    re.compile(r"^(standard|default|common|obvious) (choice|decision|approach)\b", re.I),
]


def cleanup_result(result: ExtractionResult, transcript: str = "") -> ExtractionResult:  # noqa: D103
    """Post-extraction cleanup: normalize owners, strip filler, validate dates."""
    for d in result.decisions:
        d.owner = _clean_owner(d.owner)
        d.rationale = _clean_filler(d.rationale)
        if transcript:
            d.rationale = _clean_ungrounded(d.rationale, transcript)
            d.date = _clean_date(d.date, transcript)

    for q in result.questions:
        q.owner = _clean_owner(q.owner)
        q.context = _clean_filler(q.context)
        if transcript:
            q.context = _clean_ungrounded(q.context, transcript)

    for a in result.action_items:
        a.owner = _clean_owner(a.owner)
        if transcript:
            a.deadline = _clean_date(a.deadline, transcript)

    return result


def _clean_owner(value: str) -> str:  # noqa: D103
    if not value:
        return ""
    if _VALID_OWNER_RE.match(value):
        return value
    if _BAD_OWNER_RE.search(value):
        return ""
    if value == value.lower():
        return ""
    return value


def _clean_filler(value: str) -> str:  # noqa: D103
    if not value:
        return ""
    for pattern in _FILLER_PATTERNS:
        if pattern.search(value.strip()):
            return ""
    return value


def _clean_ungrounded(value: str, transcript: str) -> str:  # noqa: D103
    if not value:
        return ""
    value_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', value))
    if not value_words:
        return value
    transcript_lower = transcript.lower()
    grounded = sum(1 for w in value_words if w in transcript_lower)
    if grounded / len(value_words) < 0.6:
        return ""
    return value


def _clean_date(value: str, transcript: str) -> str:  # noqa: D103
    if not value:
        return ""
    return value if value in transcript else ""
