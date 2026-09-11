from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, Field, StrictStr, field_validator

from orchestrator import orchestrate_question


logger = logging.getLogger("genai_sales_assistant.api")
logger.setLevel(logging.INFO)
REQUEST_ID_HEADER = "X-Request-ID"

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


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unavailable")
    failure_category = classify_technical_failure(exc)
    log_arguments = (
        request_id,
        request.url.path,
        failure_category,
        type(exc).__name__,
    )
    if failure_category == "unexpected_internal_bug":
        logger.exception(
            "api_request_failed request_id=%s path=%s category=%s error_type=%s",
            *log_arguments,
        )
    else:
        logger.error(
            "api_request_failed request_id=%s path=%s category=%s error_type=%s",
            *log_arguments,
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> Dict[str, Any]:
    request_id = request.state.request_id
    logger.info("api_request_accepted request_id=%s path=/query", request_id)
    result = orchestrate_question(payload.question)
    logger.info(
        "api_request_completed request_id=%s route=%s status=%s",
        request_id,
        result.get("route", "unknown"),
        result.get("status", "unknown"),
    )
    return result


def classify_technical_failure(exc: Exception) -> str:
    if isinstance(
        exc,
        (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError),
    ):
        return "transient_dependency_failure"
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return "non_transient_dependency_configuration_failure"
    return "unexpected_internal_bug"
