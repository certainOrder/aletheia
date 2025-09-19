from fastapi import HTTPException
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL


class OpenAIService:
    def __init__(self, api_key: str | None = None):
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY not configured")
        self.client = OpenAI(api_key=key)

    def get_response(self, prompt: str, model: str | None = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model or OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def chat(self, messages: list[dict], model: str | None = None):
        try:
            response = self.client.chat.completions.create(
                model=model or OPENAI_CHAT_MODEL,
                messages=messages,
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))