import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import dotenv
from google import genai
from google.genai import types

from llm_moses.config.settings import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_INITIAL_BACKOFF_SECONDS,
    DEFAULT_BACKOFF_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """Unified client for querying Google Gemini models for both Seeder and Mutator components."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        env_path: Optional[str] = None,
    ):
        # Resolve API key from arguments, environment, or .env files
        if not api_key:
            if env_path:
                dotenv.load_dotenv(env_path)
            else:
                # Search in current directory, parent directory, and workspace root
                search_paths = [
                        Path.cwd() / ".env",
                        Path(__file__).resolve().parent.parent.parent / ".env",
                        Path(__file__).resolve().parent.parent.parent.parent / ".env",
                        Path(__file__).resolve().parent.parent.parent.parent.parent / ".env",
                    ]
                for p in search_paths:
                    if p.is_file():
                        dotenv.load_dotenv(p)
                        break
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment, .env file, or initialization arguments."
            )

        self.api_key = api_key
        self.model_name = model_name or DEFAULT_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def generate_candidates(
        self,
        prompt: str,
        schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.7,
        n_retries: int = DEFAULT_MAX_RETRIES,
    ) -> list[str]:
        """Generate a list of candidate expressions from Gemini given a prompt.

        Handles exponential backoff, rate limits (RESOURCE_EXHAUSTED / 429),
        and automatic fallback across models.

        Args:
            prompt: Formatted prompt describing the task and grammar.
            schema: Optional JSON schema for structured outputs.
            temperature: Sampling temperature.
            n_retries: Maximum number of retry attempts on transient errors.

        Returns:
            A list of candidate MeTTa expression strings.
        """
        current_model = self.model_name
        last_error = None
        models_to_try = [current_model]
        if current_model == DEFAULT_MODEL:
            models_to_try.append(FALLBACK_MODEL)
        elif current_model == FALLBACK_MODEL:
            models_to_try.append(DEFAULT_MODEL)

        for model in models_to_try:
            for attempt in range(n_retries):
                try:
                    config_kwargs: dict[str, Any] = {
                        "temperature": temperature,
                    }
                    if schema:
                        config_kwargs["response_mime_type"] = "application/json"
                        config_kwargs["response_schema"] = schema

                    config = types.GenerateContentConfig(**config_kwargs)
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )

                    text = response.text or ""
                    candidates = self._parse_response(text)
                    if candidates:
                        return candidates
                    else:
                        logger.warning(
                            f"Model {model} returned empty candidates. Raw text: {text[:200]}"
                        )

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    logger.warning(
                        f"Attempt {attempt + 1}/{n_retries} failed for model {model}: {err_str}"
                    )

                    # Extract suggested retry delay if present in rate-limit error
                    delay = DEFAULT_INITIAL_BACKOFF_SECONDS * (DEFAULT_BACKOFF_MULTIPLIER ** attempt)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        match = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
                        if match:
                            delay = float(match.group(1)) + 1.0
                        else:
                            delay = max(delay, 8.0)
                        logger.info(f"Rate limited (429). Sleeping for {delay:.2f}s before retry...")
                        time.sleep(delay)
                    elif attempt < n_retries - 1:
                        time.sleep(delay)

            logger.warning(f"All retries failed for model {model}. Attempting fallback...")

        logger.error(f"All models exhausted. Last error: {last_error}")
        return []

    def _parse_response(self, raw_text: str) -> list[str]:
        """Parses model response text into a list of candidate expressions."""
        text = raw_text.strip()
        # Try direct JSON parsing
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item).strip() for item in data if item]
            if isinstance(data, dict):
                for key in ["candidates", "expressions", "programs", "mutants", "solutions"]:
                    if key in data and isinstance(data[key], list):
                        return [str(item).strip() for item in data[key] if item]
        except json.JSONDecodeError:
            pass

        # Extract JSON fenced block if present
        json_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return [str(item).strip() for item in data if item]
                if isinstance(data, dict):
                    for key in ["candidates", "expressions", "programs", "mutants", "solutions"]:
                        if key in data and isinstance(data[key], list):
                            return [str(item).strip() for item in data[key] if item]
            except json.JSONDecodeError:
                pass

        # Extract S-expression lines directly
        candidates = []
        for line in text.splitlines():
            line = line.strip()
            # Strip markdown list markers or numbering
            line = re.sub(r"^[\d+\-\*#\.\s]+", "", line).strip()
            if line.startswith("(") and line.endswith(")"):
                candidates.append(line)

        return candidates
    