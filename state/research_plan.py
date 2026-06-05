from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Literal, Optional
import uuid


@dataclass
class ResearchQuestion:
    question: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: Literal["pending", "in_progress", "answered", "skipped"] = "pending"
    priority: int = 1  # 1=high, 2=medium, 3=low
    findings_summary: str = ""
    sources: List[str] = field(default_factory=list)
    retry_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchQuestion":
        return cls(**data)


@dataclass
class ResearchPlan:
    """
    The agent's structured decomposition of the original query.
    Defines "done" (all questions answered/skipped), prevents loops
    (tracks what's been covered), and enables adaptive planning
    (questions can be added or skipped mid-research).
    """
    original_query: str
    questions: List[ResearchQuestion]
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    overall_status: Literal["planning", "researching", "synthesizing", "complete"] = "planning"
    iteration: int = 0  # number of mid-research plan revisions

    # --- Query helpers ---

    def pending_questions(self) -> List[ResearchQuestion]:
        return [q for q in self.questions if q.status == "pending"]

    def answered_questions(self) -> List[ResearchQuestion]:
        return [q for q in self.questions if q.status == "answered"]

    def is_complete(self) -> bool:
        return all(q.status in ("answered", "skipped") for q in self.questions)

    def get_question_by_id(self, question_id: str) -> Optional[ResearchQuestion]:
        return next((q for q in self.questions if q.id == question_id), None)

    def progress_summary(self) -> str:
        total = len(self.questions)
        answered = len(self.answered_questions())
        pending = len(self.pending_questions())
        return f"{answered}/{total} answered, {pending} pending"

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()

    # --- Serialization ---

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchPlan":
        questions = [ResearchQuestion.from_dict(q) for q in data.pop("questions")]
        return cls(questions=questions, **data)
