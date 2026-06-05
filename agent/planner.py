import json
from typing import Any, Dict

import config
from llm.client import LLMClient
from llm import prompts
from state.research_plan import ResearchPlan, ResearchQuestion


class ResearchPlanner:
    """
    Creates and revises ResearchPlans.
    Uses the LLM to decompose queries into sub-questions and adapt
    the plan when mid-research findings reveal new important angles.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def create_plan(self, query: str) -> ResearchPlan:
        print(f"\n[planner] Decomposing: {query[:80]}")
        prompt = prompts.decompose_query_prompt(query, config.MAX_QUESTIONS_PER_PLAN)
        raw = self.llm.call(prompt, system=prompts.DECOMPOSE_SYSTEM)

        try:
            questions_data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Planner returned invalid JSON: {e}\nRaw response: {raw[:200]}")

        questions = [
            ResearchQuestion(
                question=item["question"],
                priority=item.get("priority", 1),
            )
            for item in questions_data
        ]

        plan = ResearchPlan(
            original_query=query,
            questions=questions,
            overall_status="researching",
        )

        print(f"[planner] Plan created — {len(questions)} questions:")
        for q in questions:
            print(f"  [{q.id}] P{q.priority} {q.question}")

        return plan

    def revise_plan(self, plan: ResearchPlan, new_insight: str) -> Dict[str, Any]:
        """
        Optionally adds new sub-questions or skips irrelevant ones.
        Returns a revision dict; caller applies it via apply_revision().
        No-ops if plan has already been revised MAX_PLAN_REVISIONS times.
        """
        if plan.iteration >= config.MAX_PLAN_REVISIONS:
            return {"add_questions": [], "skip_question_ids": []}

        questions_status = "\n".join(
            f"[{q.id}] ({q.status}) {q.question}" for q in plan.questions
        )
        prompt = prompts.revise_plan_prompt(plan.original_query, questions_status, new_insight)
        raw = self.llm.call(prompt, system=prompts.REVISE_PLAN_SYSTEM)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"add_questions": [], "skip_question_ids": []}

    def apply_revision(self, plan: ResearchPlan, revision: Dict[str, Any]) -> None:
        """Mutates the plan in place based on a revision dict."""
        added = 0
        # Allow slight overflow beyond MAX_QUESTIONS to absorb discovered sub-questions
        hard_cap = config.MAX_QUESTIONS_PER_PLAN + 3

        for item in revision.get("add_questions", []):
            if len(plan.questions) < hard_cap:
                new_q = ResearchQuestion(
                    question=item["question"],
                    priority=item.get("priority", 2),
                )
                plan.questions.append(new_q)
                added += 1
                print(f"  [planner] + Added question: {new_q.question[:70]}")

        for qid in revision.get("skip_question_ids", []):
            q = plan.get_question_by_id(qid)
            if q and q.status == "pending":
                q.status = "skipped"
                print(f"  [planner] - Skipped: {q.question[:70]}")

        if added > 0 or revision.get("skip_question_ids"):
            plan.iteration += 1
