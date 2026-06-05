from llm.client import LLMClient
from llm import prompts
from outputs.citations import build_citation_list, format_citation_list
from state.research_session import ResearchSession


class ReportWriter:
    """
    Synthesizes all research findings into a structured final report.
    Called once, after the orchestrator has finished working through the plan.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def write_report(self, session: ResearchSession) -> str:
        plan = session.plan
        findings_store = session.findings_store

        # Build per-question findings text for the synthesis prompt
        sections = []
        for q in plan.questions:
            if q.status not in ("answered", "skipped"):
                continue
            raw_findings = findings_store.findings_text_for_question(q.id, max_chars=3000)
            sections.append(
                f"### Sub-question: {q.question}\n"
                f"Status: {q.status}\n"
                f"Evaluator summary: {q.findings_summary or '(not evaluated)'}\n\n"
                f"Raw findings:\n{raw_findings}"
            )

        findings_by_question = "\n\n---\n\n".join(sections) if sections else "(no findings)"

        # Build numbered citation list
        all_sources = findings_store.all_sources()
        citations = build_citation_list(all_sources)
        source_list = (
            "\n".join(f"[{c['index']}] {c['url']}" for c in citations)
            or "(no sources)"
        )

        # LLM synthesis call
        prompt = prompts.synthesize_report_prompt(
            query=plan.original_query,
            findings_by_question=findings_by_question,
            source_list=source_list,
        )
        report_body = self.llm.call(
            prompt,
            system=prompts.SYNTHESIZE_SYSTEM,
            max_tokens=4096,
            temperature=0.4,
        )

        # Append references section
        references = format_citation_list(citations)
        return f"{report_body}\n\n{references}" if references else report_body
