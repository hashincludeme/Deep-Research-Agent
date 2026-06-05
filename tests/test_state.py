import pytest
from memory.deduplicator import Deduplicator
from memory.findings_store import Finding, FindingsStore
from state.research_plan import ResearchPlan, ResearchQuestion
from state.research_session import ResearchSession


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_plan(n=3) -> ResearchPlan:
    questions = [ResearchQuestion(question=f"Question {i+1}", priority=1) for i in range(n)]
    return ResearchPlan(original_query="test query", questions=questions)


# ── ResearchPlan ──────────────────────────────────────────────────────────────

def test_plan_complete_when_all_answered():
    plan = make_plan()
    for q in plan.questions:
        q.status = "answered"
    assert plan.is_complete()


def test_plan_complete_with_skipped():
    plan = make_plan()
    plan.questions[0].status = "answered"
    plan.questions[1].status = "skipped"
    plan.questions[2].status = "answered"
    assert plan.is_complete()


def test_plan_not_complete_with_pending():
    plan = make_plan()
    plan.questions[0].status = "answered"
    assert not plan.is_complete()


def test_plan_progress_summary():
    plan = make_plan(3)
    plan.questions[0].status = "answered"
    assert "1/3" in plan.progress_summary()


def test_plan_serialization_roundtrip():
    plan = make_plan()
    plan.questions[0].status = "answered"
    plan.questions[0].findings_summary = "Found X and Y"
    restored = ResearchPlan.from_dict(plan.to_dict())
    assert restored.original_query == plan.original_query
    assert len(restored.questions) == 3
    assert restored.questions[0].status == "answered"
    assert restored.questions[0].findings_summary == "Found X and Y"
    assert restored.session_id == plan.session_id


def test_plan_get_question_by_id():
    plan = make_plan()
    qid = plan.questions[1].id
    found = plan.get_question_by_id(qid)
    assert found is not None
    assert found.question == "Question 2"


# ── FindingsStore ─────────────────────────────────────────────────────────────

def make_finding(question_id: str, url: str) -> Finding:
    return Finding(question_id=question_id, source_url=url, title="T", raw_content="content " * 10)


def test_findings_store_add_and_retrieve():
    store = FindingsStore()
    store.add(make_finding("q1", "https://a.com"))
    store.add(make_finding("q1", "https://b.com"))
    store.add(make_finding("q2", "https://c.com"))
    assert len(store.get_for_question("q1")) == 2
    assert len(store.get_for_question("q2")) == 1
    assert len(store.get_all()) == 3


def test_findings_store_all_sources_unique():
    store = FindingsStore()
    store.add(make_finding("q1", "https://a.com"))
    store.add(make_finding("q2", "https://a.com"))  # duplicate URL
    store.add(make_finding("q2", "https://b.com"))
    sources = store.all_sources()
    assert len(sources) == 2
    assert "https://a.com" in sources


def test_findings_store_serialization_roundtrip():
    store = FindingsStore()
    store.add(make_finding("q1", "https://example.com"))
    restored = FindingsStore.from_dict(store.to_dict())
    findings = restored.get_for_question("q1")
    assert len(findings) == 1
    assert findings[0].source_url == "https://example.com"


# ── Deduplicator ──────────────────────────────────────────────────────────────

def test_deduplicator_url_tracking():
    d = Deduplicator()
    assert not d.is_url_visited("https://example.com")
    d.mark_url_visited("https://example.com")
    assert d.is_url_visited("https://example.com")
    assert not d.is_url_visited("https://other.com")


def test_deduplicator_query_case_insensitive():
    d = Deduplicator()
    d.mark_query_executed("Best Python Libraries")
    assert d.is_query_executed("best python libraries")
    assert d.is_query_executed("  Best Python Libraries  ")


def test_deduplicator_serialization_roundtrip():
    d = Deduplicator()
    d.mark_url_visited("https://a.com")
    d.mark_query_executed("test query")
    restored = Deduplicator.from_dict(d.to_dict())
    assert restored.is_url_visited("https://a.com")
    assert restored.is_query_executed("test query")
    assert not restored.is_url_visited("https://b.com")


# ── ResearchSession ───────────────────────────────────────────────────────────

def test_session_serialization_roundtrip():
    plan = make_plan()
    session = ResearchSession(plan=plan)
    session.total_searches = 7
    session.findings_store.add(make_finding(plan.questions[0].id, "https://x.com"))
    session.deduplicator.mark_url_visited("https://x.com")

    restored = ResearchSession.from_dict(session.to_dict())
    assert restored.plan.original_query == plan.original_query
    assert restored.total_searches == 7
    findings = restored.findings_store.get_for_question(plan.questions[0].id)
    assert len(findings) == 1
    assert restored.deduplicator.is_url_visited("https://x.com")
