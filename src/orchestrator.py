from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from generate import generate_grounded_answer
from retrieve import ingest_and_build_index, retrieve_relevant_chunks
from router import route_question
from structured_lookup import lookup_compatibility
from ingest import PROJECT_ROOT


def orchestrate_question(
    question: str,
    *,
    project_root: Path | str = PROJECT_ROOT,
    as_of_date: date | None = None,
    generation_client: Any | None = None,
    retrieval_index: Any | None = None,
) -> dict[str, Any]:
    route_plan = route_question(question, project_root=project_root)
    root = Path(project_root)

    if route_plan["route"] == "structured_lookup":
        execution_result = _execute_structured_route(
            question=question,
            route_plan=route_plan,
            project_root=root,
            as_of_date=as_of_date,
        )
        return {
            "question": question,
            "route": route_plan["route"],
            "status": execution_result["status"],
            "domains": route_plan["domains"],
            "result": execution_result["result"],
            "subresults": [],
            "missing_information": execution_result["missing_information"],
            "review_required": execution_result["review_required"],
            "router_reason": route_plan["reason"],
        }

    if route_plan["route"] == "rag":
        execution_result = _execute_rag_route(
            intent=question,
            domain_ids=route_plan["domains"],
            authoritative_source_ids=route_plan.get("authoritative_source_ids", []),
            project_root=root,
            as_of_date=as_of_date,
            generation_client=generation_client,
            retrieval_index=retrieval_index,
        )
        return {
            "question": question,
            "route": route_plan["route"],
            "status": execution_result["status"],
            "domains": route_plan["domains"],
            "result": execution_result["result"],
            "subresults": [],
            "missing_information": [],
            "review_required": execution_result["review_required"],
            "router_reason": route_plan["reason"],
        }

    if route_plan["route"] == "needs_review":
        return {
            "question": question,
            "route": route_plan["route"],
            "status": "needs_review",
            "domains": route_plan["domains"],
            "result": {
                "message": "Automated resolution is not safe under the current governance rules.",
                "reason": route_plan["reason"],
            },
            "subresults": [],
            "missing_information": route_plan.get("missing_information", []),
            "review_required": True,
            "router_reason": route_plan["reason"],
        }

    if route_plan["route"] == "composite":
        subresults = []
        shared_index = retrieval_index
        for subroute in route_plan["subroutes"]:
            if subroute["route"] == "structured_lookup":
                subresults.append(
                    _execute_structured_subroute(
                        subroute=subroute,
                        project_root=root,
                        as_of_date=as_of_date,
                    )
                )
                continue

            if subroute["route"] == "rag":
                subresult = _execute_rag_subroute(
                    subroute=subroute,
                    project_root=root,
                    as_of_date=as_of_date,
                    generation_client=generation_client,
                    retrieval_index=shared_index,
                )
                if shared_index is None and "retrieval_index" in subresult:
                    shared_index = subresult.pop("retrieval_index")
                subresults.append(subresult)
                continue

            subresults.append(
                {
                    "intent": subroute["intent"],
                    "domain": subroute["domain"],
                    "route": subroute["route"],
                    "status": "needs_review",
                    "result": {
                        "message": "This subroute cannot be executed automatically."
                    },
                    "missing_information": subroute.get("missing_information", []),
                    "review_required": True,
                }
            )

        return {
            "question": question,
            "route": route_plan["route"],
            "status": _derive_composite_status(subresults),
            "domains": route_plan["domains"],
            "result": None,
            "subresults": subresults,
            "missing_information": sorted(
                {
                    item
                    for subresult in subresults
                    for item in subresult.get("missing_information", [])
                }
            ),
            "review_required": any(subresult["review_required"] for subresult in subresults),
            "router_reason": route_plan["reason"],
        }

    raise ValueError(f"Unsupported route returned by router: {route_plan['route']}")


def _execute_structured_route(
    *,
    question: str,
    route_plan: dict[str, Any],
    project_root: Path,
    as_of_date: date | None,
) -> dict[str, Any]:
    execution = route_plan.get("execution") or {}
    missing_information = list(execution.get("missing_information", []))
    if missing_information:
        return {
            "status": "clarification_needed",
            "result": {
                "message": "Structured lookup requires additional information before execution.",
                "required_parameters": missing_information,
            },
            "missing_information": missing_information,
            "review_required": False,
        }

    lookup_result = lookup_compatibility(
        inverter_model=execution.get("inverter_model"),
        battery_model=execution.get("battery_model"),
        firmware_version=execution.get("firmware_version"),
        region=execution.get("region"),
        project_root=project_root,
        as_of_date=as_of_date,
    )
    return {
        "status": _map_structured_lookup_status(lookup_result["status"]),
        "result": lookup_result,
        "missing_information": [],
        "review_required": False,
    }


def _execute_structured_subroute(
    *,
    subroute: dict[str, Any],
    project_root: Path,
    as_of_date: date | None,
) -> dict[str, Any]:
    execution = subroute.get("execution") or {}
    missing_information = list(subroute.get("missing_information", []))
    if missing_information:
        return {
            "intent": subroute["intent"],
            "domain": subroute["domain"],
            "route": subroute["route"],
            "status": "clarification_needed",
            "result": {
                "message": "Structured lookup requires additional information before execution.",
                "required_parameters": missing_information,
            },
            "missing_information": missing_information,
            "review_required": False,
        }

    lookup_result = lookup_compatibility(
        inverter_model=execution.get("inverter_model"),
        battery_model=execution.get("battery_model"),
        firmware_version=execution.get("firmware_version"),
        region=execution.get("region"),
        project_root=project_root,
        as_of_date=as_of_date,
    )
    return {
        "intent": subroute["intent"],
        "domain": subroute["domain"],
        "route": subroute["route"],
        "status": _map_structured_lookup_status(lookup_result["status"]),
        "result": lookup_result,
        "missing_information": [],
        "review_required": False,
    }


def _execute_rag_subroute(
    *,
    subroute: dict[str, Any],
    project_root: Path,
    as_of_date: date | None,
    generation_client: Any | None,
    retrieval_index: Any | None,
) -> dict[str, Any]:
    execution_result = _execute_rag_route(
        intent=subroute["intent"],
        domain_ids=[subroute["domain"]],
        authoritative_source_ids=subroute.get("authoritative_source_ids", []),
        project_root=project_root,
        as_of_date=as_of_date,
        generation_client=generation_client,
        retrieval_index=retrieval_index,
    )
    return {
        "intent": subroute["intent"],
        "domain": subroute["domain"],
        "route": subroute["route"],
        "status": execution_result["status"],
        "result": execution_result["result"],
        "missing_information": [],
        "review_required": execution_result["review_required"],
        **({"retrieval_index": execution_result["retrieval_index"]} if "retrieval_index" in execution_result else {}),
    }


def _execute_rag_route(
    *,
    intent: str,
    domain_ids: list[str],
    authoritative_source_ids: list[str],
    project_root: Path,
    as_of_date: date | None,
    generation_client: Any | None,
    retrieval_index: Any | None,
) -> dict[str, Any]:
    index = retrieval_index or ingest_and_build_index(project_root=project_root, as_of_date=as_of_date)
    retrieval_result = retrieve_relevant_chunks(
        query=intent,
        index=index,
        authoritative_source_ids=authoritative_source_ids,
        include_diagnostics=True,
    )
    retrieved_chunks = retrieval_result["chunks"]

    if not retrieved_chunks:
        return {
            "status": "insufficient_evidence",
            "result": {
                "answer": "The available governed evidence does not safely support an answer.",
                "citations": [],
                "used_evidence": [],
                "insufficient_evidence": True,
                "unresolved_points": [],
                "answer_completeness": "none",
                "unresolved_dependencies": [
                    {
                        "type": "missing_evidence",
                        "subject": "answer_support",
                        "authority_source_ids": authoritative_source_ids,
                        "reason": "The available governed evidence does not safely support a substantive answer.",
                        "blocking": True,
                    }
                ],
                "validation_status": "validated",
                "validation_issues": [],
                "retrieved_chunks": [],
                "retrieval_diagnostics": {
                    "requested_authoritative_source_ids": retrieval_result["requested_authoritative_source_ids"],
                    "retrieved_authoritative_source_ids": retrieval_result["retrieved_authoritative_source_ids"],
                    "authority_gap": retrieval_result["authority_gap"],
                },
            },
            "review_required": False,
            "retrieval_index": index,
        }

    generation_result = generate_grounded_answer(
        question=intent,
        retrieved_chunks=retrieved_chunks,
        client=generation_client,
    )
    status = _map_generation_status(generation_result)
    return {
        "status": status,
        "result": {
            **generation_result,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_diagnostics": {
                "requested_authoritative_source_ids": retrieval_result["requested_authoritative_source_ids"],
                "retrieved_authoritative_source_ids": retrieval_result["retrieved_authoritative_source_ids"],
                "authority_gap": retrieval_result["authority_gap"],
            },
        },
        "review_required": status == "needs_review",
        "retrieval_index": index,
    }


def _map_structured_lookup_status(lookup_status: str) -> str:
    if lookup_status == "invalid_input":
        return "clarification_needed"
    return "completed"


def _map_generation_status(generation_result: dict[str, Any]) -> str:
    if generation_result.get("validation_status") == "failed":
        return "generation_failed"
    if generation_result.get("validation_status") == "needs_review":
        return "needs_review"
    answer_completeness = generation_result.get("answer_completeness")
    if answer_completeness == "partial":
        return "partial"
    if answer_completeness == "none":
        return "insufficient_evidence"
    return "completed"


def _derive_composite_status(subresults: list[dict[str, Any]]) -> str:
    statuses = {subresult["status"] for subresult in subresults}
    if "clarification_needed" in statuses:
        return "clarification_needed"
    if "needs_review" in statuses:
        return "needs_review"
    if "generation_failed" in statuses:
        return "generation_failed"
    if "partial" in statuses:
        return "partial"
    if "insufficient_evidence" in statuses:
        return "insufficient_evidence"
    return "completed"
