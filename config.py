import os
from dotenv import load_dotenv

load_dotenv()

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Research settings
MAX_QUESTIONS_PER_PLAN = int(os.getenv("MAX_QUESTIONS_PER_PLAN", "7"))
MAX_SOURCES_PER_QUESTION = int(os.getenv("MAX_SOURCES_PER_QUESTION", "3"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_PLAN_REVISIONS = int(os.getenv("MAX_PLAN_REVISIONS", "2"))
MAX_RETRIES_PER_QUESTION = int(os.getenv("MAX_RETRIES_PER_QUESTION", "2"))
CONTENT_MAX_CHARS = int(os.getenv("CONTENT_MAX_CHARS", "8000"))

# Checkpointing
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")
