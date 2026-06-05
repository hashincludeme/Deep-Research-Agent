# Ariadne — Deep Research Agent

A production-grade autonomous research agent that takes a natural language query, decomposes it into a structured research plan, searches the web, scrapes sources, evaluates findings, and synthesizes a cited report — with full crash recovery via checkpointing.

---

## What It Does

```
python main.py "What are the main drivers of enterprise SaaS churn?"
```

Ariadne will:
1. Decompose the query into 4–7 specific sub-questions
2. Search the web for each sub-question (via Tavily)
3. Scrape and clean content from the top sources
4. Evaluate whether each question is sufficiently answered
5. Adaptively revise the plan if findings reveal new important angles
6. Synthesize a structured, cited research report

---

## Project Structure

```
ariadne/
├── main.py                      # CLI entry point
├── config.py                    # All settings from environment variables
│
├── state/                       # Data contracts — defined first, used by everything
│   ├── research_plan.py         # ResearchPlan + ResearchQuestion dataclasses
│   ├── research_session.py      # Full session state: plan + findings + dedup
│   └── checkpointing.py         # Serialize/deserialize session to JSON on disk
│
├── memory/                      # What the agent accumulates during research
│   ├── findings_store.py        # Scraped content keyed by question ID
│   └── deduplicator.py          # Tracks visited URLs and executed queries
│
├── llm/                         # LLM abstraction — swap models here, nowhere else
│   ├── client.py                # Anthropic SDK wrapper with retry + backoff
│   └── prompts.py               # All prompt templates in one place
│
├── tools/                       # I/O only — no reasoning, no state
│   ├── web_search.py            # Tavily API search
│   ├── web_scraper.py           # Fetch + clean full page text
│   ├── document_reader.py       # Read local .txt, .md, .pdf files
│   └── registry.py              # Maps tool names to callables
│
├── agent/                       # Reasoning layer — uses all layers below
│   ├── planner.py               # Decomposes queries, revises plans mid-research
│   ├── evaluator.py             # Decides if a question is answered
│   └── orchestrator.py          # Main agent loop: plan → search → evaluate → synthesize
│
├── outputs/                     # Final artifact generation
│   ├── report_writer.py         # LLM synthesis of all findings into a report
│   └── citations.py             # URL deduplication + numbered citation list
│
└── tests/
    ├── test_state.py            # Plan/session serialization, FindingsStore, Deduplicator
    ├── test_tools.py            # Search/scraper graceful degradation, registry
    └── test_planner.py          # Plan creation, revision logic, max-iterations guard
```

### Layer dependencies (what can import what)

```
config           ← no dependencies
state/           ← config
memory/          ← config
llm/             ← config, state/
tools/           ← config
agent/           ← all layers above
outputs/         ← llm/, memory/, state/
main.py          ← agent/orchestrator only
```

`agent/orchestrator.py` is the only file that imports from all layers.
All other layers are orchestrator-unaware and independently testable.

---

## Architecture

### Plan-and-Execute (not ReAct)

Ariadne first decomposes the query into a `ResearchPlan` before searching anything. This gives the agent:
- A definition of "done" (all questions answered or skipped)
- Loop prevention (tracks what's been covered)
- Adaptive planning (new sub-questions can be added mid-research)

A pure ReAct loop follows the last search result and meanders. A plan gives the agent a structured objective.

### Stateful with checkpointing

Every meaningful step saves the full `ResearchSession` to disk (`sessions/<id>.json`). If the agent crashes mid-run — a 20-minute research job at step 45 — resume with:

```bash
python main.py --resume <session_id>
```

The session includes the plan, all findings, and deduplication state. Nothing is re-fetched.

### Separation of concerns

| Layer | Does | Does not |
|---|---|---|
| `state/` | Define data shapes | Make decisions |
| `tools/` | I/O (HTTP, file reads) | Reason about content |
| `memory/` | Store and retrieve findings | Search or scrape |
| `llm/` | Call the LLM | Know about research plans |
| `agent/` | Orchestrate all layers | Do I/O directly |
| `outputs/` | Generate artifacts | Manage agent loop |

---

## Setup

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `TAVILY_API_KEY` | Yes | Tavily search API ([free tier](https://tavily.com)) |
| `LLM_MODEL` | No | Default: `claude-sonnet-4-6` |
| `MAX_QUESTIONS_PER_PLAN` | No | Default: `7` |
| `MAX_SOURCES_PER_QUESTION` | No | Default: `3` |

---

## Usage

```bash
# Basic research
python main.py "What are the main causes of enterprise SaaS churn?"

# Save report to file
python main.py "Explain the competitive landscape of vector databases" --output report.md

# Resume a crashed or paused session
python main.py --resume a3f2b1c4

# List all saved sessions
python main.py --list-sessions
```

### Sample terminal output

```
Ariadne — Deep Research Agent
============================================================
Query : What are the main causes of enterprise SaaS churn?
Model : claude-sonnet-4-6
============================================================

[planner] Decomposing: What are the main causes of enterprise SaaS churn?
[planner] Plan created — 6 questions:
  [a1b2c3d4] P1 What product-related factors drive SaaS churn?
  [e5f6g7h8] P1 How does onboarding quality affect churn rates?
  [i9j0k1l2] P1 What role does customer success play in retention?
  [m3n4o5p6] P2 How do pricing and contract structures affect churn?
  [q7r8s9t0] P2 What data signals predict churn before it happens?
  [u1v2w3x4] P3 How do enterprise churn rates compare across industries?

[orchestrator] Session: 7f3a9b2c-...
[orchestrator] Progress: 0/6 answered, 6 pending

[orchestrator] Researching [a1b2c3d4]: What product-related factors drive SaaS churn?
  [search] 5 result(s)
  [scraper] https://www.gainsight.com/blog/top-reasons-saas... (7,842 chars)
  [scraper] https://hbr.org/2023/saas-churn-product... (6,201 chars)
  [evaluator] answered=True confidence=0.88
  [orchestrator] Progress: 1/6 answered, 5 pending
...

[orchestrator] Research complete. Synthesizing report...
[orchestrator] Done. Searches: 6 | Pages scraped: 14 | Session: 7f3a9b2c-...
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

27 tests covering state serialization, tool graceful degradation, and planner logic. No API keys required — all external calls are mocked.

---

## Extending Ariadne

**Add a new tool:** Define a function in `tools/`, register it in `tools/registry.py`, call it via `get_tool("name")` in the orchestrator.

**Change the LLM:** Edit `config.py` → `LLM_MODEL`. The `LLMClient` in `llm/client.py` handles the rest.

**Add a new output format:** Add a writer to `outputs/` that takes a `ResearchSession` and returns a string. Call it from the orchestrator after `report_writer`.

**Persist sessions to a database:** Replace `state/checkpointing.py` with a Postgres/Redis backend — no other files need to change.
