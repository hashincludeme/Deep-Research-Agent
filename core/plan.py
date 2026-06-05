"""
thena/core/plan.py

ResearchPlan: the authoritative state object for one Thena research session.

Lives outside conversation history so it:
- Survives context window limits (injected into system prompt, never truncated)
- Can be serialized to disk for crash recovery and session resumption
- Provides O(1) state queries instead of conversational reconstruction
- Makes state transitions explicit and testable via immutable with_*() methods
"""

from __future__ import annotations

import dataclasses
import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

# ── Type aliases ──────────────────────────────────────────────────────────────

StepID = str
FindingID = str
GapID = str
SourceURL = str


# ── Enums ─────────────────────────────────────────────────────────────────────

class PlanStatus(str, Enum):
    PLANNING     = "planning"      # Initial decomposition in progress
    RESEARCHING  = "researching"   # Actively working through steps
    REPLANNING   = "replanning"    # Revising plan based on mid-research findings
    SYNTHESIZING = "synthesizing"  # All steps done, writing report
    COMPLETE     = "complete"      # Report generated and saved
    FAILED       = "failed"        # Unrecoverable error


class StepStatus(str, Enum):
    PENDING     = "pending"      # Not yet started
    IN_PROGRESS = "in_progress"  # Currently being researched
    COMPLETE    = "complete"     # Sufficient findings obtained
    SKIPPED     = "skipped"      # Could not be answered; recorded as a gap
    FAILED      = "failed"       # Repeated failures with no usable findings


class SourceCredibility(str, Enum):
    HIGH    = "high"     # Peer-reviewed, government, established institutions
    MEDIUM  = "medium"   # Reputable press, known industry sources
    LOW     = "low"      # Blogs, forums, unverified claims
    UNKNOWN = "unknown"  # Not yet assessed


class GapSeverity(str, Enum):
    CRITICAL = "critical"  # Prevents answering the core question
    MODERATE = "moderate"  # Reduces confidence but does not block synthesis
    MINOR    = "minor"     # Would strengthen the report but is not essential


# ── Nested Pydantic models ────────────────────────────────────────────────────

class Step(BaseModel):
    """
    One sub-question the agent must answer as part of the research plan.
    Steps are the atomic unit of work — the planner creates them,
    the orchestrator executes them, the evaluator marks them complete.
    """
    model_config = ConfigDict(frozen=True)

    id: StepID = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description=(
            "Short unique identifier for this step. "
            "Used to reference it in findings, completions, and gap records."
        ),
    )
    question: str = Field(
        description=(
            "The specific, searchable sub-question this step must answer. "
            "Should be precise enough to form a direct web search query."
        ),
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current lifecycle state of this step.",
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Execution priority: 1=must answer (foundational), "
            "2=important context, 3=nice to have. "
            "Orchestrator always picks the lowest-numbered pending step first."
        ),
    )
    depends_on: list[StepID] = Field(
        default_factory=list,
        description=(
            "Step IDs that must reach COMPLETE or SKIPPED before this step can start. "
            "Used to enforce research ordering when later sub-questions build on earlier findings."
        ),
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of times this step has been attempted and returned insufficient findings. "
            "When retry_count >= MAX_RETRIES_PER_QUESTION, the step is skipped and a Gap is recorded."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Why the planner included this step — the reasoning behind the decomposition choice. "
            "Injected into the system prompt via to_xml() to give the LLM context for each step."
        ),
    )


class Source(BaseModel):
    """
    A web page or document fetched and read during research.
    Every URL that is accessed is recorded as a Source, regardless of whether
    its content ultimately contributed to a Finding. This enables deduplication
    and provides a complete audit trail of what was read.
    """
    model_config = ConfigDict(frozen=True)

    url: SourceURL = Field(description="Full URL of the fetched source.")
    title: str = Field(default="", description="Page title or document name.")
    domain: str = Field(
        default="",
        description="Extracted domain without 'www.' prefix (e.g. 'reuters.com').",
    )
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this source was fetched.",
    )
    content_hash: str = Field(
        default="",
        description=(
            "First 16 hex characters of SHA-256 of the scraped content. "
            "Used to detect duplicate content across different URLs."
        ),
    )
    char_count: int = Field(
        default=0,
        description="Number of characters in the scraped content after cleaning.",
    )
    scrape_succeeded: bool = Field(
        default=True,
        description=(
            "False if full page scraping failed and only the search snippet was stored. "
            "The report writer notes snippet-only sources with lower confidence."
        ),
    )

    @classmethod
    def from_scrape(
        cls,
        url: str,
        title: str,
        content: str,
        scrape_succeeded: bool = True,
    ) -> "Source":
        """
        Convenience constructor that derives domain and content_hash from raw inputs.
        Called by the orchestrator immediately after scraping a page.
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            url=url,
            title=title,
            domain=domain,
            content_hash=content_hash,
            char_count=len(content),
            scrape_succeeded=scrape_succeeded,
        )


class SourceAnalysis(BaseModel):
    """
    The evaluator's per-source assessment of credibility and extracted claims.
    Produced during the evaluation step, stored inside the Finding it informed.
    Allows the report writer to cite sources with calibrated confidence levels.
    """
    model_config = ConfigDict(frozen=True)

    source_url: SourceURL = Field(description="URL of the source being analyzed.")
    step_id: StepID = Field(description="Which step this analysis was produced for.")
    credibility: SourceCredibility = Field(
        default=SourceCredibility.UNKNOWN,
        description=(
            "Credibility tier based on domain reputation, author attribution, "
            "and verifiability of specific claims made."
        ),
    )
    key_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Specific factual claims extracted from this source that are directly "
            "relevant to the step's question. Verbatim or close paraphrase."
        ),
    )
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Known reliability limitations: article is dated, content is behind a paywall "
            "and snippet was used, source has a known editorial bias, claims are anecdotal, etc."
        ),
    )


class Finding(BaseModel):
    """
    A distilled, evaluator-verified answer to one research step.
    Produced after the LLM evaluates scraped content and confirms sufficiency
    (confidence >= threshold). Multiple findings can exist per step if replanning
    added the step back after an initial partial answer.
    """
    model_config = ConfigDict(frozen=True)

    id: FindingID = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this finding.",
    )
    step_id: StepID = Field(
        description="The step this finding answers. Foreign key into ResearchPlan.steps.",
    )
    content: str = Field(
        description=(
            "The actual research finding — a 2-4 sentence synthesis of what was discovered. "
            "Written by the evaluator LLM call, not copied verbatim from sources."
        ),
    )
    source_urls: list[SourceURL] = Field(
        default_factory=list,
        description=(
            "URLs of the sources that support this finding, in order of relevance. "
            "Used by citations.py to build the numbered references section."
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Evaluator's confidence score that this finding adequately answers the step (0.0–1.0). "
            "Scores below 0.6 trigger a retry; scores below 0.4 after MAX_RETRIES become gaps."
        ),
    )
    source_analyses: list[SourceAnalysis] = Field(
        default_factory=list,
        description=(
            "Per-source credibility analysis for each source that contributed to this finding. "
            "Enables the report writer to flag low-credibility sources explicitly."
        ),
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this finding was recorded by the evaluator.",
    )


class Gap(BaseModel):
    """
    A known limitation in the research — a sub-question that could not be answered,
    or an aspect of the topic that was unreachable due to source unavailability,
    paywalls, or insufficient public information.

    Gaps are surfaced explicitly in the report's 'Limitations' section
    rather than being silently omitted. A report that acknowledges its gaps
    is more trustworthy than one that pretends to be complete.
    """
    model_config = ConfigDict(frozen=True)

    id: GapID = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this gap.",
    )
    description: str = Field(
        description=(
            "What specific information is missing and why it matters to the original research question. "
            "Written to be human-readable in the final report's limitations section."
        ),
    )
    related_step_ids: list[StepID] = Field(
        default_factory=list,
        description=(
            "IDs of steps that were skipped or returned insufficient findings, "
            "producing this gap. A gap can result from multiple failed steps."
        ),
    )
    severity: GapSeverity = Field(
        default=GapSeverity.MODERATE,
        description=(
            "How much this gap affects the quality of the final report. "
            "CRITICAL gaps prevent synthesis from confidently answering the core question."
        ),
    )
    search_queries_tried: list[str] = Field(
        default_factory=list,
        description=(
            "The search queries that were executed before declaring this a gap. "
            "Included for auditability — shows the agent made a genuine effort."
        ),
    )


class Contradiction(BaseModel):
    """
    A factual conflict between two sources on the same topic.
    Contradictions are recorded and surfaced in the report rather than
    silently resolved. The agent does not pick a winner — it presents
    both claims and lets the reader assess credibility.
    """
    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this contradiction.",
    )
    description: str = Field(
        description=(
            "What the two sources disagree about, and why both claims are plausible given "
            "their respective contexts. Written for inclusion in the report's limitations section."
        ),
    )
    source_a_url: SourceURL = Field(description="URL of the first source making a claim.")
    source_b_url: SourceURL = Field(description="URL of the second source making a conflicting claim.")
    claim_a: str = Field(
        default="",
        description="The specific claim from source A, verbatim or close paraphrase.",
    )
    claim_b: str = Field(
        default="",
        description="The conflicting claim from source B, verbatim or close paraphrase.",
    )
    related_step_ids: list[StepID] = Field(
        default_factory=list,
        description="Steps where this contradiction was encountered during research.",
    )


# ── ResearchPlan ──────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ResearchPlan:
    """
    Authoritative, immutable state object for one Thena research session.

    FROZEN — why:
      State transitions are explicit. Every change produces a NEW ResearchPlan
      via a with_*() method. The old plan is never modified. This means:
      - State mutations are visible and auditable (old_plan → new_plan)
      - No accidental in-place corruption across the orchestrator's loop
      - Every plan version is independently testable
      - Frozen dataclasses are hashable (usable as dict keys, in sets, for caching)

    SEPARATE FROM CONVERSATION HISTORY — why:
      Conversation history grows unboundedly and gets truncated. This plan is
      injected into the system prompt via to_xml() on every LLM call so the
      model always has the complete, current research state regardless of
      how many messages have accumulated.
    """

    # ── Core identity ──────────────────────────────────────────────────────

    question: str
    """The original research query that initiated this plan."""

    session_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )
    """Unique identifier for this session. Used as the filename for checkpointing."""

    status: PlanStatus = PlanStatus.PLANNING
    """Overall lifecycle state of the research session."""

    created_at: datetime = dataclasses.field(default_factory=datetime.utcnow)
    """UTC timestamp when this plan was first created by the planner."""

    last_replan_at: Optional[datetime] = None
    """UTC timestamp of the most recent plan revision. None if the plan has never been revised."""

    # ── Research structure ─────────────────────────────────────────────────

    steps: tuple[Step, ...] = dataclasses.field(default_factory=tuple)
    """
    All sub-questions the agent must answer, as an ordered tuple.
    Tuple (not list) because the dataclass is frozen — a list field can be
    mutated in place even when the dataclass is frozen; a tuple cannot.
    """

    completed: tuple[StepID, ...] = dataclasses.field(default_factory=tuple)
    """
    IDs of steps that have reached COMPLETE, SKIPPED, or FAILED status.
    Maintained as a separate tuple for O(1) membership checks without
    scanning all steps each time.
    """

    # ── Accumulated knowledge ──────────────────────────────────────────────

    findings: tuple[Finding, ...] = dataclasses.field(default_factory=tuple)
    """
    All findings verified by the evaluator during this session.
    Each finding references its parent step via step_id.
    """

    gaps: tuple[Gap, ...] = dataclasses.field(default_factory=tuple)
    """
    Known limitations — steps that produced no usable findings after all retries.
    Surfaced in the final report's Gaps and Limitations section.
    """

    sources: tuple[Source, ...] = dataclasses.field(default_factory=tuple)
    """
    Every source URL accessed during this session, in order of first access.
    Used by with_source() deduplication and by citations.py to build references.
    """

    contradictions: tuple[Contradiction, ...] = dataclasses.field(default_factory=tuple)
    """
    Factual conflicts found between sources. Recorded by the evaluator when
    two sources make incompatible claims about the same topic.
    """

    # ── Resource tracking ──────────────────────────────────────────────────

    tool_calls_used: int = 0
    """
    Running count of all tool calls made (search + scrape + evaluate LLM calls).
    Incremented via with_tool_call(). Used for budget enforcement and telemetry.
    """

    token_budget_remaining: int = 100_000
    """
    Approximate remaining token budget for LLM calls in this session.
    Decremented by with_tool_call() and with_replan() based on estimated costs.
    When this reaches 0, the orchestrator moves directly to synthesis.
    """

    # ── Computed properties ────────────────────────────────────────────────

    @property
    def pending_steps(self) -> tuple[Step, ...]:
        """All steps not yet started, sorted by priority ascending (1 first)."""
        pending = [s for s in self.steps if s.status == StepStatus.PENDING]
        return tuple(sorted(pending, key=lambda s: s.priority))

    @property
    def is_complete(self) -> bool:
        """True when every step has reached a terminal status (COMPLETE, SKIPPED, FAILED)."""
        terminal = {StepStatus.COMPLETE, StepStatus.SKIPPED, StepStatus.FAILED}
        return all(s.status in terminal for s in self.steps)

    @property
    def progress(self) -> str:
        """Human-readable progress string for terminal output and logging."""
        total = len(self.steps)
        done = len(self.completed)
        return f"{done}/{total} complete, {total - done} pending"

    # ── XML serialization ──────────────────────────────────────────────────

    def to_xml(self) -> str:
        """
        Serializes the full plan state to an indented XML string.

        This XML is injected into the LLM system prompt on every agent call,
        giving the model a complete, structured view of current research state
        without relying on conversation history.

        XML over JSON because:
        - Claude handles XML well (its own output format uses XML-like tags)
        - Hierarchical structure maps naturally to nested research state
        - ElementTree escapes special characters in findings/questions automatically
        - The <tag> format is more readable than JSON in a long system prompt

        The output fits inside a <research_plan>...</research_plan> block
        that the orchestrator wraps in its system prompt template.
        """
        root = ET.Element("research_plan")

        # Identity and status
        ET.SubElement(root, "session_id").text = self.session_id
        ET.SubElement(root, "question").text = self.question
        ET.SubElement(root, "status").text = self.status.value

        # Progress summary
        prog = ET.SubElement(root, "progress")
        ET.SubElement(prog, "steps_total").text = str(len(self.steps))
        ET.SubElement(prog, "steps_complete").text = str(len(self.completed))
        ET.SubElement(prog, "steps_pending").text = str(len(self.pending_steps))
        ET.SubElement(prog, "summary").text = self.progress

        # Steps
        steps_el = ET.SubElement(root, "steps")
        for step in self.steps:
            s_el = ET.SubElement(steps_el, "step")
            s_el.set("id", step.id)
            s_el.set("status", step.status.value)
            s_el.set("priority", str(step.priority))
            ET.SubElement(s_el, "question").text = step.question
            if step.rationale:
                ET.SubElement(s_el, "rationale").text = step.rationale
            if step.retry_count > 0:
                ET.SubElement(s_el, "retry_count").text = str(step.retry_count)
            if step.depends_on:
                deps_el = ET.SubElement(s_el, "depends_on")
                for dep_id in step.depends_on:
                    ET.SubElement(deps_el, "step_id").text = dep_id

        # Findings
        if self.findings:
            findings_el = ET.SubElement(root, "findings")
            for f in self.findings:
                f_el = ET.SubElement(findings_el, "finding")
                f_el.set("id", f.id)
                f_el.set("step_id", f.step_id)
                f_el.set("confidence", f"{f.confidence:.2f}")
                ET.SubElement(f_el, "content").text = f.content
                if f.source_urls:
                    srcs_el = ET.SubElement(f_el, "sources")
                    for url in f.source_urls:
                        ET.SubElement(srcs_el, "url").text = url
                if f.source_analyses:
                    analyses_el = ET.SubElement(f_el, "source_analyses")
                    for sa in f.source_analyses:
                        sa_el = ET.SubElement(analyses_el, "analysis")
                        sa_el.set("url", sa.source_url)
                        sa_el.set("credibility", sa.credibility.value)
                        if sa.key_claims:
                            claims_el = ET.SubElement(sa_el, "key_claims")
                            for claim in sa.key_claims:
                                ET.SubElement(claims_el, "claim").text = claim
                        if sa.limitations:
                            lims_el = ET.SubElement(sa_el, "limitations")
                            for lim in sa.limitations:
                                ET.SubElement(lims_el, "limitation").text = lim

        # Gaps
        if self.gaps:
            gaps_el = ET.SubElement(root, "gaps")
            for g in self.gaps:
                g_el = ET.SubElement(gaps_el, "gap")
                g_el.set("id", g.id)
                g_el.set("severity", g.severity.value)
                ET.SubElement(g_el, "description").text = g.description
                if g.related_step_ids:
                    related_el = ET.SubElement(g_el, "related_steps")
                    for sid in g.related_step_ids:
                        ET.SubElement(related_el, "step_id").text = sid
                if g.search_queries_tried:
                    queries_el = ET.SubElement(g_el, "search_queries_tried")
                    for q in g.search_queries_tried:
                        ET.SubElement(queries_el, "query").text = q

        # Contradictions
        if self.contradictions:
            contras_el = ET.SubElement(root, "contradictions")
            for c in self.contradictions:
                c_el = ET.SubElement(contras_el, "contradiction")
                c_el.set("id", c.id)
                ET.SubElement(c_el, "description").text = c.description
                ET.SubElement(c_el, "claim_a").text = c.claim_a
                ET.SubElement(c_el, "claim_b").text = c.claim_b
                srcs_el = ET.SubElement(c_el, "sources")
                ET.SubElement(srcs_el, "source_a").text = c.source_a_url
                ET.SubElement(srcs_el, "source_b").text = c.source_b_url

        # Resource budget
        budget_el = ET.SubElement(root, "budget")
        ET.SubElement(budget_el, "token_budget_remaining").text = str(self.token_budget_remaining)
        ET.SubElement(budget_el, "tool_calls_used").text = str(self.tool_calls_used)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=False)

    # ── Immutable update methods ───────────────────────────────────────────

    def with_finding(self, finding: Finding) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with the finding appended to findings.

        Does NOT mark the associated step complete — that is a deliberate
        separation of concerns. The evaluator calls with_finding() first
        (to record what was found), then calls with_step_complete() separately
        (to mark the step done). This allows multiple findings per step
        and keeps the recording and the lifecycle decision independent.
        """
        return dataclasses.replace(self, findings=(*self.findings, finding))

    def with_gap(self, gap: Gap) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with the gap appended to gaps.

        Called by the orchestrator when a step exhausts all retries without
        producing a confident finding. The gap is also displayed in the
        final report's Limitations section so the reader knows what the
        agent could not determine.
        """
        return dataclasses.replace(self, gaps=(*self.gaps, gap))

    def with_step_complete(
        self,
        step_id: StepID,
        new_status: StepStatus = StepStatus.COMPLETE,
    ) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with the given step marked at new_status
        and its ID added to the completed tuple.

        Raises ValueError for unknown step_id — fails loudly rather than
        silently producing a plan with a phantom completion record.

        Uses Step.model_copy(update=...) rather than constructing a new Step
        from scratch, so all other step fields (question, priority, rationale,
        retry_count, depends_on) are preserved unchanged.

        The completed tuple is maintained separately from steps so that
        'is this step done?' is an O(1) set-like check rather than an O(n)
        scan of all steps.
        """
        if not any(s.id == step_id for s in self.steps):
            raise ValueError(
                f"Step ID '{step_id}' not found in this plan. "
                f"Available IDs: {[s.id for s in self.steps]}"
            )
        updated_steps = tuple(
            s.model_copy(update={"status": new_status}) if s.id == step_id else s
            for s in self.steps
        )
        return dataclasses.replace(
            self,
            steps=updated_steps,
            completed=(*self.completed, step_id),
        )

    def with_source(self, source: Source) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with the source appended, if not already present.

        Deduplicates by URL — if the same URL appears in search results for
        two different steps, it is stored only once. This prevents the sources
        tuple from accumulating duplicates that inflate the citation count.
        """
        if any(s.url == source.url for s in self.sources):
            return self
        return dataclasses.replace(self, sources=(*self.sources, source))

    def with_contradiction(self, contradiction: Contradiction) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with the contradiction appended.

        Called by the evaluator when two sources in the same finding window
        make incompatible factual claims. The orchestrator does not try to
        resolve contradictions — it records them for the report writer to
        surface transparently.
        """
        return dataclasses.replace(
            self, contradictions=(*self.contradictions, contradiction)
        )

    def with_replan(
        self,
        new_steps: list[Step],
        token_cost: int = 0,
        new_status: PlanStatus = PlanStatus.RESEARCHING,
    ) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with additional steps appended.

        Used when mid-research findings reveal important sub-questions not
        in the original plan. Called by the planner after apply_revision().

        token_cost is deducted from token_budget_remaining to account for
        the LLM call that produced the revision.

        last_replan_at is stamped so the orchestrator can track revision
        frequency — too-frequent replanning signals a poorly scoped original
        question and may warrant early termination.
        """
        return dataclasses.replace(
            self,
            steps=(*self.steps, *new_steps),
            status=new_status,
            last_replan_at=datetime.utcnow(),
            token_budget_remaining=max(0, self.token_budget_remaining - token_cost),
        )

    def with_tool_call(self, token_cost: int = 0) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with tool_calls_used incremented
        and token_budget_remaining decremented by token_cost.

        Called by the orchestrator after EVERY external call:
        web search, page scrape, evaluator LLM call, and planner LLM call.
        This keeps a running budget that the orchestrator can check before
        each step — if token_budget_remaining reaches 0, it skips remaining
        steps and moves directly to synthesis with whatever has been found.
        """
        return dataclasses.replace(
            self,
            tool_calls_used=self.tool_calls_used + 1,
            token_budget_remaining=max(0, self.token_budget_remaining - token_cost),
        )

    def with_status(self, status: PlanStatus) -> "ResearchPlan":
        """
        Returns a new ResearchPlan with the overall status updated.

        Used by the orchestrator to mark lifecycle transitions:
          PLANNING → RESEARCHING  (after create_plan)
          RESEARCHING → REPLANNING (briefly, during plan revision)
          REPLANNING → RESEARCHING (after revision is applied)
          RESEARCHING → SYNTHESIZING (all steps terminal)
          SYNTHESIZING → COMPLETE (report written)
          Any state → FAILED (unrecoverable error)
        """
        return dataclasses.replace(self, status=status)
