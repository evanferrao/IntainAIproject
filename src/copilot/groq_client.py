"""Real Groq LLM API Client with graceful degradation and latency tracking."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.logger import logger


def _load_env_file():
    """Lightweight .env loader without external dependencies."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")


_load_env_file()


class GroqClient:
    """Official Groq API client for natural-language loan review conversation."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        _load_env_file()
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "").strip()
        self.model = model if model is not None else (os.environ.get("GROQ_MODEL", "").strip() or self.DEFAULT_MODEL)
        self._client = None

    def is_available(self) -> bool:
        """Returns True if a non-empty GROQ_API_KEY is configured."""
        return bool(self.api_key)

    def get_configured_model(self) -> str:
        """Returns the configured model name."""
        return self.model

    def _get_client(self):
        """Lazily initializes the Groq client."""
        if self._client is None and self.is_available():
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1200
    ) -> Dict[str, Any]:
        """Sends an explicit natural-language query to Groq and tracks latency."""
        if not self.is_available():
            return {
                "success": False,
                "response": None,
                "model": self.model,
                "latency_seconds": 0.0,
                "error": "GROQ_API_KEY is not configured in .env. Live conversational LLM is unavailable. Non-LLM ML analytics, predictions, explainability, and reviewer dispositions remain fully operational."
            }

        start_time = time.perf_counter()
        try:
            client = self._get_client()
            logger.info(f"Calling Groq API (model: {self.model})...")
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            latency = round(time.perf_counter() - start_time, 3)
            reply = completion.choices[0].message.content

            logger.info(f"Groq API call succeeded in {latency}s.")
            return {
                "success": True,
                "response": reply,
                "model": self.model,
                "latency_seconds": latency,
                "error": None
            }
        except Exception as e:
            latency = round(time.perf_counter() - start_time, 3)
            error_msg = f"Groq API error ({type(e).__name__}): {str(e)}"
            logger.error(f"Groq call failed after {latency}s: {error_msg}")
            return {
                "success": False,
                "response": None,
                "model": self.model,
                "latency_seconds": latency,
                "error": error_msg
            }
