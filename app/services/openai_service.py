import hashlib
import random
from typing import Any, cast

from fastapi import HTTPException
from openai import OpenAI

from app.config import DEV_FALLBACKS, OPENAI_API_KEY, OPENAI_CHAT_MODEL


class OpenAIService:
    def __init__(self, api_key: str | None = None):
        key = api_key or OPENAI_API_KEY
        self.api_key = key
        self.client = OpenAI(api_key=key) if key else None

    def get_response(self, prompt: str, model: str | None = None) -> str:
        try:
            if self.client is None or DEV_FALLBACKS:
                # Local fallback when no API key provided
                return self._local_response(prompt)
            response = self.client.chat.completions.create(
                model=cast(Any, (model or OPENAI_CHAT_MODEL)),
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            if DEV_FALLBACKS:
                # Fallback to local response in dev when remote fails
                return self._local_response(prompt)
            raise HTTPException(status_code=500, detail=str(e))

    def chat(self, messages: list[dict[str, Any]], model: str | None = None):
        try:
            if self.client is None or DEV_FALLBACKS:
                return self._local_chat_response(messages, model)
            # The SDK expects a union of specific message param types;
            # our runtime dicts are compatible.
            response = self.client.chat.completions.create(
                model=cast(Any, (model or OPENAI_CHAT_MODEL)),
                messages=cast(Any, messages),
            )
            return response
        except Exception as e:
            if DEV_FALLBACKS:
                return self._local_chat_response(messages, model)
            raise HTTPException(status_code=500, detail=str(e))

    # --- Local fallback helpers ---
    def _local_response(self, prompt: str) -> str:
        # Extremely simple deterministic echo-like response for dev
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        prefixes = [
            "Draft: ",
            "Thought: ",
            "Note: ",
            "Answer: ",
        ]
        return prefixes[rng.randrange(len(prefixes))] + prompt

    def _local_chat_response(self, messages: list[dict], model: str | None):
        # Mimic OpenAI ChatCompletion response structure minimally
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        content = self._local_response(last_user)
        return {
            "model": model or OPENAI_CHAT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
