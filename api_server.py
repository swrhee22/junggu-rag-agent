from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_agent

app = FastAPI(title = 'Junggu Guide Agent API')


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    route: str


@app.get("/")
def root():
    return {"message": "Junggu Guide Agent API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, sources, route = run_agent(req.question)
    return ChatResponse(
        answer = answer,
        sources = sources,
        route = route
    )