from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictStr, field_validator

from orchestrator import orchestrate_question


RouteValue = Literal["structured_lookup", "rag", "needs_review", "composite"]
StatusValue = Literal[
    "completed",
    "clarification_needed",
    "partial",
    "needs_review",
    "insufficient_evidence",
    "generation_failed",
]


class QueryRequest(BaseModel):
    question: StrictStr

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value


class SubrouteResponse(BaseModel):
    intent: str
    domain: str
    route: str
    status: str
    result: Optional[Dict[str, Any]]
    missing_information: List[str] = Field(default_factory=list)
    review_required: bool


class QueryResponse(BaseModel):
    question: str
    route: RouteValue
    status: StatusValue
    domains: List[str]
    result: Optional[Dict[str, Any]]
    subresults: List[SubrouteResponse] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    review_required: bool
    router_reason: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


app = FastAPI()


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> Dict[str, Any]:
    return orchestrate_question(payload.question)
