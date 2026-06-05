#!/usr/bin/env python3
"""
Ariadne — Deep Research Agent

Usage:
  python main.py "your research query"
  python main.py "your query" --output report.md
  python main.py --resume <session_id>
  python main.py --list-sessions
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Ariadne: Production-grade deep research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?", help="Research query to investigate")
    parser.add_argument("--resume", metavar="SESSION_ID", help="Resume a previous session")
    parser.add_argument("--list-sessions", action="store_true", help="List all saved sessions")
    parser.add_argument("--output", metavar="FILE", help="Save final report to a file")
    args = parser.parse_args()

    if args.list_sessions:
        from state.checkpointing import list_sessions
        sessions = list_sessions()
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Saved sessions ({len(sessions)}):")
            for sid in sessions:
                print(f"  {sid}")
        return

    if not args.query and not args.resume:
        parser.print_help()
        sys.exit(1)

    import config

    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env or export it.")
        sys.exit(1)

    if not config.TAVILY_API_KEY:
        print("WARNING: TAVILY_API_KEY not set. Web search will return no results.")

    from agent.orchestrator import ResearchOrchestrator

    query = args.query or "(resuming)"
    print(f"\nAriadne — Deep Research Agent")
    print(f"{'=' * 60}")
    print(f"Query : {query}")
    print(f"Model : {config.LLM_MODEL}")
    if args.resume:
        print(f"Resume: {args.resume}")
    print(f"{'=' * 60}")

    orchestrator = ResearchOrchestrator()
    report = orchestrator.run(query=query, resume_session_id=args.resume)

    print(f"\n{'=' * 60}")
    print("FINAL REPORT")
    print(f"{'=' * 60}\n")
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
