import json
from pathlib import Path
from typing import List

import config
from state.research_session import ResearchSession


def _session_path(session_id: str) -> Path:
    sessions_dir = Path(config.SESSIONS_DIR)
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir / f"{session_id}.json"


def save_session(session: ResearchSession) -> None:
    """Serializes the full ResearchSession to disk as JSON."""
    path = _session_path(session.plan.session_id)
    session.touch()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)


def load_session(session_id: str) -> ResearchSession:
    """Deserializes a session from disk. Raises FileNotFoundError if not found."""
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"No session checkpoint found: {session_id}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ResearchSession.from_dict(data)


def list_sessions() -> List[str]:
    """Returns all session IDs that have been checkpointed."""
    sessions_dir = Path(config.SESSIONS_DIR)
    if not sessions_dir.exists():
        return []
    return [p.stem for p in sorted(sessions_dir.glob("*.json"))]
