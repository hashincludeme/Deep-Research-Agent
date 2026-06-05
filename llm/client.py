import time
from typing import Optional

import anthropic
import config


class LLMClient:
    """
    Thin wrapper around the Anthropic SDK.
    Single responsibility: make LLM calls with retry and exponential backoff.
    All other layers call this — swap the underlying model here without
    touching planner, evaluator, or report_writer.
    """

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retries: int = 3,
    ) -> str:
        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

        kwargs = {
            "model": config.LLM_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        for attempt in range(retries):
            try:
                response = self._client.messages.create(**kwargs)
                return response.content[0].text

            except anthropic.RateLimitError:
                if attempt == retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  [llm] Rate limit — retrying in {wait}s...")
                time.sleep(wait)

            except anthropic.APIError as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  [llm] API error ({e}) — retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError("LLM call failed after all retries")
