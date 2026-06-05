from typing import Optional

import config
from agent.evaluator import ResearchEvaluator
from agent.planner import ResearchPlanner
from llm.client import LLMClient
from memory.findings_store import Finding
from outputs.report_writer import ReportWriter
from state.checkpointing import load_session, save_session
from state.research_plan import ResearchQuestion
from state.research_session import ResearchSession
from tools.registry import get_tool


class ResearchOrchestrator:
    """
    Main agent loop. Coordinates:
      planner → search → scrape → evaluate → (revise plan) → synthesize

    Supports crash recovery via session_id — pass an existing session_id
    to resume from the last checkpoint instead of starting over.

    Architecture:
    - Orchestrator is the ONLY file that imports from all layers.
    - All other layers are unaware of the orchestrator.
    - This makes every layer independently testable.
    """

    def __init__(self):
        llm = LLMClient()
        self.planner = ResearchPlanner(llm)
        self.evaluator = ResearchEvaluator(llm)
        self.report_writer = ReportWriter(llm)
        self.search = get_tool("web_search")
        self.scrape = get_tool("web_scraper")

    def run(self, query: str, resume_session_id: Optional[str] = None) -> str:
        # ── Load or create session ────────────────────────────────────────────
        if resume_session_id:
            print(f"\n[orchestrator] Resuming session: {resume_session_id}")
            session = load_session(resume_session_id)
        else:
            plan = self.planner.create_plan(query)
            session = ResearchSession(plan=plan)
            save_session(session)

        plan = session.plan
        print(f"\n[orchestrator] Session: {plan.session_id}")
        print(f"[orchestrator] Progress: {plan.progress_summary()}")

        # ── Main research loop ────────────────────────────────────────────────
        while not plan.is_complete():
            question = self._next_question(session)
            if not question:
                break

            print(f"\n[orchestrator] Researching [{question.id}]: {question.question}")
            question.status = "in_progress"

            # Search + scrape
            self._research_question(session, question)

            # Evaluate completeness
            evaluation = self.evaluator.evaluate_question(question, session.findings_store)
            print(
                f"  [evaluator] answered={evaluation['is_answered']} "
                f"confidence={evaluation['confidence']:.2f}"
            )

            if evaluation["is_answered"]:
                question.status = "answered"
                question.findings_summary = evaluation["summary"]
                question.sources = [
                    f.source_url
                    for f in session.findings_store.get_for_question(question.id)
                ]

                # Revise plan if the finding surfaces new important angles
                if evaluation["summary"] and plan.iteration < config.MAX_PLAN_REVISIONS:
                    revision = self.planner.revise_plan(plan, evaluation["summary"])
                    if revision.get("add_questions") or revision.get("skip_question_ids"):
                        self.planner.apply_revision(plan, revision)

            else:
                question.retry_count += 1
                if question.retry_count >= config.MAX_RETRIES_PER_QUESTION:
                    gaps = ", ".join(evaluation.get("gaps", []))
                    question.status = "skipped"
                    question.findings_summary = f"Insufficient data. Gaps: {gaps}"
                    print(f"  [orchestrator] Skipped after {question.retry_count} attempt(s)")
                else:
                    question.status = "pending"
                    print(f"  [orchestrator] Will retry (attempt {question.retry_count + 1})")

            print(f"  [orchestrator] Progress: {plan.progress_summary()}")
            save_session(session)

        # ── Synthesize final report ───────────────────────────────────────────
        print(f"\n[orchestrator] Research complete. Synthesizing report...")
        plan.overall_status = "synthesizing"
        save_session(session)

        report = self.report_writer.write_report(session)
        session.final_report = report
        plan.overall_status = "complete"
        save_session(session)

        print(
            f"\n[orchestrator] Done. "
            f"Searches: {session.total_searches} | "
            f"Pages scraped: {session.total_pages_scraped} | "
            f"Session: {plan.session_id}"
        )
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _next_question(self, session: ResearchSession) -> Optional[ResearchQuestion]:
        """Returns the highest-priority pending question."""
        pending = session.plan.pending_questions()
        if not pending:
            return None
        return sorted(pending, key=lambda q: q.priority)[0]

    def _research_question(
        self, session: ResearchSession, question: ResearchQuestion
    ) -> None:
        """Searches and scrapes sources for one question. Mutates session in place."""
        dedup = session.deduplicator

        if dedup.is_query_executed(question.question):
            print(f"  [search] Query already executed — skipping")
            return

        results = self.search(question.question)
        dedup.mark_query_executed(question.question)
        session.total_searches += 1
        print(f"  [search] {len(results)} result(s)")

        scraped = 0
        for result in results[: config.MAX_SOURCES_PER_QUESTION]:
            url = result["url"]
            if dedup.is_url_visited(url):
                continue

            content = self.scrape(url)
            dedup.mark_url_visited(url)
            session.total_pages_scraped += 1

            if content:
                session.findings_store.add(
                    Finding(
                        question_id=question.id,
                        source_url=url,
                        title=result.get("title", ""),
                        raw_content=content,
                    )
                )
                scraped += 1
                print(f"  [scraper] {url[:65]}... ({len(content):,} chars)")

            elif result.get("content"):
                # Fall back to the Tavily search snippet if full scrape fails
                session.findings_store.add(
                    Finding(
                        question_id=question.id,
                        source_url=url,
                        title=result.get("title", ""),
                        raw_content=result["content"],
                        relevance_note="search snippet (scrape failed)",
                    )
                )
                scraped += 1

        print(f"  [orchestrator] {scraped} source(s) added for [{question.id}]")
