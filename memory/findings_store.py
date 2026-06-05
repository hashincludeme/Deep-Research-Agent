from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List


@dataclass
class Finding:
    """One scraped page or snippet associated with a specific research question."""
    question_id: str
    source_url: str
    title: str
    raw_content: str
    relevance_note: str = ""
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        return cls(**data)


class FindingsStore:
    """
    Accumulates findings during research, keyed by question ID.
    Lives inside ResearchSession and is checkpointed with it.

    Responsibilities:
    - Store findings per question for evaluator access
    - Provide all sources for citation generation
    - Format findings as text for LLM prompts
    """

    def __init__(self):
        self._store: Dict[str, List[Finding]] = {}

    def add(self, finding: Finding) -> None:
        qid = finding.question_id
        if qid not in self._store:
            self._store[qid] = []
        self._store[qid].append(finding)

    def get_for_question(self, question_id: str) -> List[Finding]:
        return self._store.get(question_id, [])

    def get_all(self) -> List[Finding]:
        return [f for findings in self._store.values() for f in findings]

    def all_sources(self) -> List[str]:
        """Unique source URLs across all findings, in order of addition."""
        seen = set()
        sources = []
        for f in self.get_all():
            if f.source_url and f.source_url not in seen:
                seen.add(f.source_url)
                sources.append(f.source_url)
        return sources

    def findings_text_for_question(self, question_id: str, max_chars: int = 6000) -> str:
        """Returns findings as a single text block for LLM prompts."""
        findings = self.get_for_question(question_id)
        if not findings:
            return "(no findings yet)"
        parts = []
        for i, f in enumerate(findings):
            parts.append(f"Source {i+1}: {f.title}\nURL: {f.source_url}\n{f.raw_content}")
        combined = "\n\n---\n\n".join(parts)
        return combined[:max_chars]

    def to_dict(self) -> dict:
        return {qid: [f.to_dict() for f in findings] for qid, findings in self._store.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "FindingsStore":
        store = cls()
        for qid, findings_list in data.items():
            store._store[qid] = [Finding.from_dict(f) for f in findings_list]
        return store
