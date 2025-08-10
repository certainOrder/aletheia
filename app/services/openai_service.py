
from fastapi import HTTPException
import openai
import os
from dotenv import load_dotenv


class OpenAIService:
    def __init__(self, api_key: str = None):
        # Load environment variables if not already loaded
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'aletheia_api.env'))
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = openai.OpenAI(api_key=api_key)

    def get_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))