
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os


# Load environment variables from .env file at startup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aletheia_api.env'))


from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.openai_service import OpenAIService


app = FastAPI()

# Serve static files (for chat UI)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Route for chat UI
@app.get("/chat")
def chat_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))


# Request model for OpenAI chat
class ChatRequest(BaseModel):
    prompt: str

# POST endpoint to test OpenAI integration
@app.post("/openai-chat")
async def openai_chat(request: ChatRequest):
    service = OpenAIService()
    response = service.get_response(request.prompt)
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}