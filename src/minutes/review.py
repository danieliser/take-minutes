"""Review mode — compare user intent against code changes for gap analysis."""

from __future__ import annotations

import json
import logging

from minutes.changes_parse import parse_changes
from minutes.intent import extract_user_prompts, summarize_intent
from minutes.models import ReviewItem, ReviewResult

logger = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT = """\
You are analyzing a coding session. Compare what the user asked for (intent) \
against what was actually changed (code changes).

Classify each sub-goal as "covered" (evidence in changes) or "gap" (no matching change).
Flag any changes not tied to a stated goal as "unasked".
Score alignment from 0.0 (nothing matched) to 1.0 (perfect match).

Return JSON matching this schema:
{
  "alignment_score": 0.85,
  "summary": "One paragraph overview of alignment",
  "covered": [{"description": "...", "evidence": "..."}],
  "gaps": [{"description": "...", "evidence": "..."}],
  "unasked": [{"description": "...", "evidence": "..."}]
}"""


def run_review(backend, file_path: str, strict: bool = False) -> ReviewResult:
    """Compare intent against changes for a session."""
    prompts = extract_user_prompts(file_path, strict=strict)
    timeline = parse_changes(file_path, strict=strict)

    if not prompts and not timeline.changes:
        return ReviewResult()

    intent = summarize_intent(backend, prompts) if prompts else None

    # Build condensed change summary (file + reasoning, no full diffs)
    change_lines = []
    for c in timeline.changes:
        reasoning = c.reasoning[:200] if c.reasoning else ""
        change_lines.append(f"- {c.action} {c.file_path}" + (f": {reasoning}" if reasoning else ""))
    changes_text = "\n".join(change_lines) if change_lines else "(no code changes)"

    # Build intent text
    if intent and intent.primary_goal:
        intent_parts = [f"Primary goal: {intent.primary_goal}"]
        if intent.sub_goals:
            intent_parts.append("Sub-goals:\n" + "\n".join(f"- {g}" for g in intent.sub_goals))
        if intent.constraints:
            intent_parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in intent.constraints))
        intent_text = "\n\n".join(intent_parts)
    elif prompts:
        intent_text = "User prompts (no structured intent available):\n" + "\n---\n".join(prompts[:5])
    else:
        intent_text = "(no user prompts found — likely a team/subagent session)"

    user_content = f"## User Intent\n{intent_text}\n\n## Code Changes ({len(timeline.changes)} total)\n{changes_text}"

    try:
        client = getattr(backend, 'client', backend)
        model = getattr(backend, 'model', "default")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content
        from minutes.extractor_chunking import extract_json_block
        json_text = extract_json_block(response_text) or response_text
        data = json.loads(json_text)

        return ReviewResult(
            alignment_score=float(data.get("alignment_score", 0.0)),
            covered=[ReviewItem(**item) for item in data.get("covered", [])],
            gaps=[ReviewItem(**item) for item in data.get("gaps", [])],
            unasked=[ReviewItem(**item) for item in data.get("unasked", [])],
            summary=data.get("summary", ""),
            intent_prompt_count=len(prompts),
            changes_count=len(timeline.changes),
        )

    except Exception as e:
        logger.warning(f"Review analysis failed: {e}")
        return ReviewResult(
            intent_prompt_count=len(prompts),
            changes_count=len(timeline.changes),
        )
