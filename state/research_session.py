from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from state.research_plan import ResearchPlan
from memory.findings_store import FindingsStore
from memory.deduplicator import Deduplicator


@dataclass
class ResearchSession:
    """
    Complete in-memory state for one research run.
    Everything the agent needs to resume from a checkpoint:
    the plan, all findings, and deduplication state.
    """
    plan: ResearchPlan
    findings_store: FindingsStore = field(default_factory=FindingsStore)
    deduplicator: Deduplicator = field(default_factory=Deduplicator)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    final_report: Optional[str] = None
    total_searches: int = 0
    total_pages_scraped: int = 0

    def touch(self) -> None:
        self.last_updated = datetime.utcnow().isoformat()
        self.plan.touch()

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "findings_store": self.findings_store.to_dict(),
            "deduplicator": self.deduplicator.to_dict(),
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "final_report": self.final_report,
            "total_searches": self.total_searches,
            "total_pages_scraped": self.total_pages_scraped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchSession":
        plan = ResearchPlan.from_dict(data.pop("plan"))
        findings_store = FindingsStore.from_dict(data.pop("findings_store", {}))
        deduplicator = Deduplicator.from_dict(data.pop("deduplicator", {}))
        return cls(
            plan=plan,
            findings_store=findings_store,
            deduplicator=deduplicator,
            **data,
        )
