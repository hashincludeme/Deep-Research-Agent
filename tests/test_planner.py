import json
import pytest
from unittest.mock import MagicMock

from agent.planner import ResearchPlanner
from state.research_plan import ResearchPlan, ResearchQuestion


def mock_llm(response) -> MagicMock:
    llm = MagicMock()
    llm.call.return_value = json.dumps(response) if not isinstance(response, str) else response
    return llm


# ── create_plan ───────────────────────────────────────────────────────────────

def test_create_plan_returns_research_plan():
    response = [
        {"question": "What is X?", "priority": 1},
        {"question": "How does Y work?", "priority": 1},
        {"question": "What are the implications?", "priority": 2},
    ]
    planner = ResearchPlanner(mock_llm(response))
    plan = planner.create_plan("Explain X and Y")

    assert isinstance(plan, ResearchPlan)
    assert len(plan.questions) == 3
    assert plan.questions[0].question == "What is X?"
    assert plan.questions[2].priority == 2
    assert plan.overall_status == "researching"


def test_create_plan_assigns_unique_ids():
    response = [{"question": f"Q{i}", "priority": 1} for i in range(5)]
    planner = ResearchPlanner(mock_llm(response))
    plan = planner.create_plan("Test query")
    ids = [q.id for q in plan.questions]
    assert len(ids) == len(set(ids)), "All question IDs should be unique"


def test_create_plan_raises_on_invalid_json():
    planner = ResearchPlanner(mock_llm("not valid json at all"))
    with pytest.raises(ValueError, match="invalid JSON"):
        planner.create_plan("Test query")


# ── revise_plan / apply_revision ──────────────────────────────────────────────

def test_apply_revision_adds_question():
    revision = {
        "add_questions": [{"question": "New angle", "priority": 2}],
        "skip_question_ids": [],
    }
    planner = ResearchPlanner(mock_llm(revision))

    plan = ResearchPlan(
        original_query="test",
        questions=[ResearchQuestion(question="Original Q", priority=1)],
    )
    planner.apply_revision(plan, revision)

    assert len(plan.questions) == 2
    assert plan.questions[1].question == "New angle"
    assert plan.iteration == 1


def test_apply_revision_skips_question():
    plan = ResearchPlan(
        original_query="test",
        questions=[
            ResearchQuestion(question="Q1"),
            ResearchQuestion(question="Q2"),
        ],
    )
    qid_to_skip = plan.questions[1].id
    revision = {"add_questions": [], "skip_question_ids": [qid_to_skip]}

    planner = ResearchPlanner(mock_llm(revision))
    planner.apply_revision(plan, revision)

    assert plan.questions[1].status == "skipped"
    assert plan.iteration == 1


def test_revise_plan_noop_at_max_iterations(monkeypatch):
    import config
    monkeypatch.setattr(config, "MAX_PLAN_REVISIONS", 2)

    plan = ResearchPlan(
        original_query="test",
        questions=[ResearchQuestion(question="Q1")],
        iteration=2,  # already at max
    )
    planner = ResearchPlanner(MagicMock())
    result = planner.revise_plan(plan, "some insight")

    assert result == {"add_questions": [], "skip_question_ids": []}
    planner.llm.call.assert_not_called()
