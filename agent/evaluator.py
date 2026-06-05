import json
from typing import Any, Dict

from llm.client import LLMClient
from llm import prompts
from memory.findings_store import FindingsStore
from state.research_plan import ResearchQuestion


class ResearchEvaluator:
    """
    Decides whether a research question has been sufficiently answered.
    Reads from FindingsStore, calls the LLM to assess completeness,
    and returns a structured evaluation dict.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate_question(
        self,
        question: ResearchQuestion,
        findings_store: FindingsStore,
    ) -> Dict[str, Any]:
        """
        Returns:
          is_answered: bool
          confidence: float 0.0–1.0
          summary: str — 2-3 sentence synthesis of what was found
          gaps: List[str] — what's still missing
        """
        findings_text = findings_store.findings_text_for_question(question.id)

        if findings_text == "(no findings yet)":
            return {
                "is_answered": False,
                "confidence": 0.0,
                "summary": "",
                "gaps": ["No sources found for this question."],
            }

        prompt = prompts.evaluate_findings_prompt(question.question, findings_text)
        raw = self.llm.call(prompt, system=prompts.EVALUATE_SYSTEM)

        try:
            result = json.loads(raw)
            # Ensure required keys exist with safe defaults
            return {
                "is_answered": bool(result.get("is_answered", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "summary": result.get("summary", ""),
                "gaps": result.get("gaps", []),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "is_answered": False,
                "confidence": 0.0,
                "summary": "",
                "gaps": ["Evaluator returned an unparseable response."],
            }
