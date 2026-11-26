"""HTTP API for the answer pipeline.

Run:  uv run --env-file .env uvicorn app.main:app --reload
Try:  http://127.0.0.1:8000/docs  (FastAPI's built-in test page)
"""

import logging  # standard library: structured log lines (CloudWatch reads these later)
import time  # standard library: perf_counter() for measuring latency

from fastapi import (
    FastAPI,  # the web framework: turns Python functions into HTTP endpoints
)
from pydantic import (  # describe the shape of JSON in and out; FastAPI validates it
    BaseModel,
    Field,
)

from app.answer import answer  # the LangGraph pipeline: question in, result dict out

log = logging.getLogger("bsp_copilot")  # our app's logger (named, so we can filter its lines)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # print INFO and up

app = FastAPI(title="BSP Circular Copilot")  # the web app; uvicorn looks for this name ("app.main:app")


class AskRequest(BaseModel):  # the JSON body /ask accepts
    question: str = Field(min_length=3, max_length=500)  # too short/long → 422 error, the pipeline never runs


class AskResponse(BaseModel):  # the JSON /ask returns
    answer: str  # the answer, or the refusal sentence
    refused: bool  # True if we refused
    reason: str  # "ok", "no passages", "model refused", "no citations", "bad citations: [...]"
    citations: list[str]  # e.g. ["BSP Circular No. 1242, p. 1"]
    latency_ms: int  # how long the whole pipeline took


@app.get("/health")  # GET /health: is the server up? (App Runner pings this later)
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")  # POST /ask with {"question": "..."}
def ask(request: AskRequest) -> AskResponse:  # plain def (not async): FastAPI runs it in a worker thread
    """Answer one question and log latency + tokens."""
    start = time.perf_counter()  # a precise stopwatch (seconds, as a float)
    result = answer(request.question)  # run the graph: retrieve → generate → check → maybe refuse
    latency_ms = int((time.perf_counter() - start) * 1000)  # seconds → milliseconds

    tokens = result["tokens"] or {}  # None when the model wasn't called (no passages)
    # One log line per question, as key=value pairs: easy to search and chart later (CloudWatch).
    # We log the outcome and cost, not the question text (it could contain personal details).
    log.info(
        "ask latency_ms=%d refused=%s reason=%r citations=%s input_tokens=%s output_tokens=%s",
        latency_ms, result["refused"], result["reason"], result["citations"],
        tokens.get("input"), tokens.get("output"),
    )
    return AskResponse(
        answer=result["answer"],
        refused=result["refused"],
        reason=result["reason"],
        citations=result["citations"],
        latency_ms=latency_ms,
    )
