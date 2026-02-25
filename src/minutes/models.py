"""Pydantic models for structured knowledge extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """A decision made during the meeting."""

    summary: str = Field(description="What was decided")
    owner: str = Field(default="", description="Who made it")
    rationale: str = Field(default="", description="Why this choice")
    date: str = Field(default="", description="When")


class Idea(BaseModel):
    """An idea or suggestion discussed."""

    title: str = Field(description="Short idea name")
    description: str = Field(default="", description="What is it")
    category: str = Field(default="suggestion", description="problem|opportunity|suggestion")


class Question(BaseModel):
    """An open question or topic."""

    text: str = Field(description="The question")
    context: str = Field(default="", description="Why it matters")
    owner: str = Field(default="", description="Who answers")


class ActionItem(BaseModel):
    """Something to do after the meeting."""

    description: str = Field(description="What to do")
    owner: str = Field(default="", description="Who owns it")
    deadline: str = Field(default="", description="When")


class Concept(BaseModel):
    """A key concept discussed."""

    name: str = Field(description="Concept name")
    definition: str = Field(default="", description="What it is")


class Term(BaseModel):
    """A technical term or abbreviation."""

    term: str = Field(description="The word or abbreviation")
    definition: str = Field(default="", description="What it means")
    context: str = Field(default="", description="Where mentioned")


class ExtractionResult(BaseModel):
    """The complete result of knowledge extraction from a transcript."""

    decisions: list[Decision] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    terms: list[Term] = Field(default_factory=list)
    tldr: str = Field(default="", description="2-3 sentence summary")


# --- v0.4.0: Layered extraction models ---


class CodeChange(BaseModel):
    """A single code change from an Edit or Write tool use."""

    sequence: int
    file_path: str
    action: str = Field(description="edit or write")
    old_content: str = ""
    new_content: str = ""
    reasoning: str = ""
    tool_use_id: str = ""


class ChangeTimeline(BaseModel):
    """Ordered sequence of code changes in a session."""

    changes: list[CodeChange] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    total_edits: int = 0
    total_writes: int = 0


class ToolCall(BaseModel):
    """A single tool invocation with context."""

    sequence: int
    tool_name: str
    tool_use_id: str = ""
    input_summary: str = ""
    reasoning: str = ""


class ToolStats(BaseModel):
    """Aggregate tool usage statistics for a session."""

    total_calls: int = 0
    by_tool: dict[str, int] = Field(default_factory=dict)
    by_file: dict[str, int] = Field(default_factory=dict)
    edit_count: int = 0
    write_count: int = 0
    read_count: int = 0
    bash_count: int = 0
    search_count: int = 0
    message_count: int = 0
    user_prompt_count: int = 0
    assistant_turn_count: int = 0
    calls: list[ToolCall] = Field(default_factory=list)


class IntentSummary(BaseModel):
    """LLM-generated summary of session intent from user prompts."""

    primary_goal: str = ""
    sub_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    prompt_count: int = 0


class ReviewItem(BaseModel):
    """A single item in a review gap analysis."""

    description: str = ""
    evidence: str = ""


class ReviewResult(BaseModel):
    """Gap analysis comparing user intent against code changes."""

    alignment_score: float = 0.0
    covered: list[ReviewItem] = Field(default_factory=list)
    gaps: list[ReviewItem] = Field(default_factory=list)
    unasked: list[ReviewItem] = Field(default_factory=list)
    summary: str = ""
    intent_prompt_count: int = 0
    changes_count: int = 0
