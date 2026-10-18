from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# criterion -> weight (sum = 100)
CRITERIA_WEIGHTS: dict[str, int] = {
    "problem_clarity": 15,
    "innovation": 20,
    "technical_depth": 25,
    "completeness": 10,
    "impact_viability": 20,
    "presentation_quality": 10,
}
assert sum(CRITERIA_WEIGHTS.values()) == 100

CRITERIA_LABELS = {
    "problem_clarity": "Problem Clarity",
    "innovation": "Innovation",
    "technical_depth": "Technical Depth",
    "completeness": "Completeness",
    "impact_viability": "Impact & Viability",
    "presentation_quality": "Presentation Quality",
}


class CriterionScore(BaseModel):
    """LLM-facing schema: all fields required (provider strict-mode safe)."""

    criterion: str = Field(description="Exact criterion id from the rubric.")
    score: float = Field(description="Score from 0 to 10.")
    justification: str = Field(description="1-3 sentences justifying the score.")
    evidence: list[str] = Field(
        description="Short verbatim quotes (max 25 words each) copied from the submission. Empty list if none."
    )

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(10.0, float(v)))


class AgentReport(BaseModel):
    criteria: list[CriterionScore]
    strengths: list[str]
    weaknesses: list[str]
    summary: str = Field(description="2-4 sentence assessment from this agent's perspective.")


class FinalEvaluation(BaseModel):
    project_title: str = Field(description="Project name from the submission, or 'Unknown'.")
    summary: str = Field(description="3-5 sentence overall assessment.")
    criteria: list[CriterionScore]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]


class EvidenceQuote(BaseModel):
    """One evidence quote cited by an agent, checked verbatim against the document."""

    criterion: str
    quote: str
    agent: str
    verified: bool


class DocumentEvaluationResult(BaseModel):
    """Final, code-assembled result for one document."""

    file_name: str
    file_type: str
    status: str  # completed | partial | failed | skipped
    project_title: str = ""
    total_score: Optional[float] = None
    verdict: str = ""
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    criterion_justifications: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_checked: int = 0
    evidence_verified: int = 0
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)  # strict-scoring caps/penalties applied in code
    char_count: int = 0
    truncated: bool = False
    elapsed_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)


VERDICT_THRESHOLDS = (("Excellent", 85), ("Strong", 70), ("Promising", 55))


def verdict_for(total: float) -> str:
    for name, floor in VERDICT_THRESHOLDS:
        if total >= floor:
            return name
    return "Needs Improvement"
