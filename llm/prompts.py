"""
All prompt templates in one place.
Nothing is hardcoded in planner.py, evaluator.py, or report_writer.py —
all prompt strings live here so they can be reviewed and tuned independently.
"""

# ── Planner ──────────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = (
    "You are a research planner. Return only valid JSON, no markdown fences, no explanation."
)


def decompose_query_prompt(query: str, max_questions: int) -> str:
    return f"""Decompose this research query into {max_questions} or fewer specific, answerable sub-questions.

Query: {query}

Each sub-question must:
- Be answerable through web research
- Cover a distinct aspect of the original query
- Be specific enough to search for directly

Return a JSON array (no other text):
[
  {{"question": "specific sub-question", "priority": 1}},
  ...
]

Priority scale: 1 = must answer, 2 = important context, 3 = nice to have.
Order from most fundamental to most specific.
Return ONLY valid JSON."""


REVISE_PLAN_SYSTEM = (
    "You are a research planner adapting a plan mid-research. "
    "Return only valid JSON, no markdown fences, no explanation."
)


def revise_plan_prompt(original_query: str, questions_status: str, new_insight: str) -> str:
    return f"""Review a research plan mid-execution. Based on a new finding, decide if the plan needs updating.

Original query: {original_query}

Current questions and status:
{questions_status}

New insight from research:
{new_insight}

Should any new sub-questions be added, or any existing ones skipped?

Return a JSON object:
{{
  "add_questions": [{{"question": "...", "priority": 1}}],
  "skip_question_ids": ["id1"]
}}

Only add questions that are genuinely important for answering the original query.
If no changes needed: {{"add_questions": [], "skip_question_ids": []}}
Return ONLY valid JSON."""


# ── Evaluator ─────────────────────────────────────────────────────────────────

EVALUATE_SYSTEM = (
    "You are a research evaluator. Return only valid JSON, no markdown fences, no explanation."
)


def evaluate_findings_prompt(question: str, findings_text: str) -> str:
    return f"""Evaluate whether this research question has been sufficiently answered by the findings below.

Question: {question}

Findings:
{findings_text}

Return a JSON object:
{{
  "is_answered": true or false,
  "confidence": 0.0 to 1.0,
  "summary": "2-3 sentence summary of what was found",
  "gaps": ["what is still missing or unclear"]
}}

Set is_answered=true only if findings directly address the question with reasonable depth.
A partial answer with low confidence should be is_answered=false.
Return ONLY valid JSON."""


# ── Report writer ─────────────────────────────────────────────────────────────

SYNTHESIZE_SYSTEM = (
    "You are a research analyst. Write clear, structured, professional prose. "
    "Use markdown headers. Do not add preamble like 'Here is your report'."
)


def synthesize_report_prompt(
    query: str, findings_by_question: str, source_list: str
) -> str:
    return f"""Write a comprehensive research report answering the following query.

Original query: {query}

Research findings (organized by sub-question):
{findings_by_question}

Sources for citation:
{source_list}

Structure the report with:
1. A direct 2-3 sentence answer to the original query at the top
2. ## sections for each major theme (not one per sub-question — synthesize across them)
3. Inline citations using [1], [2] etc. matching the source list above
4. A ## Gaps and Limitations section if any sub-questions were skipped or poorly covered
5. A ## Key Takeaways section with 3-5 bullet points

Write 600-1200 words. Be comprehensive but not repetitive.
Do not add a references section — that will be appended separately."""
