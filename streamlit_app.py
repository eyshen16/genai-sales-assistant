from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
HTTP_TIMEOUT_SECONDS = 120
HEALTH_TIMEOUT_SECONDS = 3
PRIVACY_NOTICE = "Prototype using synthetic product data. Do not enter personal, confidential, or customer information."
EXAMPLE_QUESTIONS = [
    "Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2?",
    "What is required for PV-surplus charging with ChargeOne 11?",
    "Can I add HomeCell 15 to my VE Hybrid 8 system, and will that affect my warranty?",
]
REQUIRED_QUERY_RESPONSE_FIELDS = {
    "question",
    "route",
    "status",
    "domains",
    "result",
    "subresults",
    "missing_information",
    "review_required",
    "router_reason",
}


def call_health() -> bool:
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=HEALTH_TIMEOUT_SECONDS)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except requests.RequestException:
        return False


def call_query(question: str) -> requests.Response:
    return requests.post(
        QUERY_ENDPOINT,
        json={"question": question},
        timeout=HTTP_TIMEOUT_SECONDS,
    )


def parse_query_response(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except (requests.JSONDecodeError, ValueError) as exc:
        raise ValueError("Backend response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Backend response must be a JSON object.")
    if not REQUIRED_QUERY_RESPONSE_FIELDS.issubset(payload):
        raise ValueError("Backend response did not match the expected API envelope.")
    return payload


def render_technical_error(message: str, response: requests.Response | None = None) -> None:
    st.error(message)
    if response is not None:
        request_id = response.headers.get("X-Request-ID")
        if request_id:
            st.caption(f"Request reference: `{request_id}`")


def render_status_banner(status: str, review_required: bool) -> None:
    if status == "completed":
        st.success("Answer available from the current governed system.")
    elif status == "partial":
        st.warning("Partial answer: some useful guidance is available, but part of the request remains unresolved.")
    elif status == "clarification_needed":
        st.warning("More information is needed before the system can complete this request.")
    elif status == "needs_review":
        st.warning("Automated resolution is not considered safe for this request and review is required.")
    elif status == "insufficient_evidence":
        st.info("The current governed evidence does not safely support a substantive answer.")
    elif status == "generation_failed":
        st.error("The system could not produce a valid grounded answer for this request.")

    if review_required and status != "needs_review":
        st.warning("Additional review is recommended for at least part of this result.")


def render_primary_result(route: str, status: str, result: Dict[str, Any] | None) -> None:
    if route == "structured_lookup":
        if result:
            st.subheader("Result", anchor=False)
            st.write(result.get("message", "No structured result message available."))
            if result.get("status"):
                st.caption(f"Structured result status: `{result['status']}`")
        return

    if route == "needs_review":
        if result:
            st.subheader("Result", anchor=False)
            st.write(result.get("message", "Automated resolution is not safe under the current governance rules."))
            if result.get("reason"):
                st.caption(result["reason"])
        return

    if route == "rag":
        if result:
            st.subheader("Answer", anchor=False)
            st.write(result.get("answer", "No answer text was returned."))
        return


def render_missing_information(result: Dict[str, Any] | None, missing_information: List[str]) -> None:
    required_parameters = []
    if isinstance(result, dict):
        required_parameters = result.get("required_parameters", []) or []

    missing = list(dict.fromkeys(list(missing_information) + list(required_parameters)))
    if not missing:
        return

    st.subheader("Missing Information", anchor=False)
    for item in missing:
        st.write(f"- `{item}`")


def render_citations(result: Dict[str, Any]) -> None:
    citations = result.get("citations", []) or []
    retrieved_chunks = result.get("retrieved_chunks", []) or []
    if not citations and not retrieved_chunks:
        return

    st.subheader("Evidence", anchor=False)

    if citations:
        st.markdown("**Citations**")
        for citation in citations:
            evidence_id = citation.get("evidence_id", "?")
            source_id = citation.get("source_id", "?")
            title = citation.get("title", "Unknown source")
            section = citation.get("section", "Unknown section")
            st.write(f"- `{evidence_id}` · `{source_id}` · {title} — {section}")

    if retrieved_chunks:
        with st.expander("Retrieved source details"):
            for index, chunk in enumerate(retrieved_chunks, start=1):
                label = f"{index}. {chunk.get('source_id', '?')} — {chunk.get('title', 'Unknown title')} / {chunk.get('section', 'Unknown section')}"
                with st.expander(label):
                    st.write(chunk.get("text", ""))


def render_unresolved_dependencies(result: Dict[str, Any]) -> None:
    dependencies = result.get("unresolved_dependencies", []) or []
    if not dependencies:
        return

    st.subheader("Unresolved Dependencies", anchor=False)
    for dependency in dependencies:
        subject = dependency.get("subject", "Unspecified dependency")
        reason = dependency.get("reason", "")

        st.write(f"- **{subject}**")
        if reason:
            st.write(reason)


def render_technical_details(payload: Dict[str, Any]) -> None:
    with st.expander("Technical Details"):
        st.markdown("**Top-level metadata**")
        st.write(
            {
                "route": payload.get("route"),
                "status": payload.get("status"),
                "domains": payload.get("domains"),
                "review_required": payload.get("review_required"),
                "router_reason": payload.get("router_reason"),
                "missing_information": payload.get("missing_information"),
            }
        )

        result = payload.get("result")
        if isinstance(result, dict):
            structured_details = {
                key: result.get(key)
                for key in ("status", "required_firmware", "matched_rule", "source_reference")
                if key in result
            }
            if structured_details:
                st.markdown("**Structured details**")
                st.write(structured_details)

            rag_details = {
                key: result.get(key)
                for key in (
                    "validation_status",
                    "validation_issues",
                    "answer_completeness",
                    "insufficient_evidence",
                    "used_evidence",
                    "unresolved_dependencies",
                    "retrieval_diagnostics",
                )
                if key in result
            }
            if rag_details:
                st.markdown("**RAG details**")
                st.write(rag_details)


def render_rag_or_structured_section(title: str, route: str, status: str, result: Dict[str, Any] | None, missing_information: List[str], review_required: bool) -> None:
    st.subheader(title, anchor=False)
    render_status_banner(status, review_required)
    render_primary_result(route, status, result)
    render_missing_information(result, missing_information)

    if route == "rag" and isinstance(result, dict):
        render_unresolved_dependencies(result)
        render_citations(result)


def render_composite(payload: Dict[str, Any]) -> None:
    render_status_banner(str(payload.get("status", "")), bool(payload.get("review_required")))

    aggregated_missing = payload.get("missing_information", []) or []
    if aggregated_missing:
        st.subheader("Missing Information", anchor=False)
        for item in aggregated_missing:
            st.write(f"- `{item}`")

    st.subheader("Subresults", anchor=False)
    for index, subresult in enumerate(payload.get("subresults", []), start=1):
        route = str(subresult.get("route", ""))
        status = str(subresult.get("status", ""))
        domain = subresult.get("domain", "unknown_domain")
        intent = subresult.get("intent", f"Subresult {index}")
        result = subresult.get("result")
        missing_information = subresult.get("missing_information", []) or []
        review_required = bool(subresult.get("review_required"))

        with st.container(border=True):
            st.subheader(f"{index}. {intent}", anchor=False)
            render_status_banner(status, review_required)
            render_primary_result(route, status, result if isinstance(result, dict) else None)
            render_missing_information(result if isinstance(result, dict) else None, missing_information)

            if route == "rag" and isinstance(result, dict):
                render_unresolved_dependencies(result)
                render_citations(result)

            with st.expander("Technical Details"):
                st.write(
                    {
                        "domain": domain,
                        "route": route,
                        "status": status,
                        **subresult,
                    }
                )


def render_payload(payload: Dict[str, Any]) -> None:
    route = str(payload.get("route", ""))
    status = str(payload.get("status", ""))
    result = payload.get("result")
    missing_information = payload.get("missing_information", []) or []
    review_required = bool(payload.get("review_required"))

    if route == "composite":
        render_composite(payload)
        render_technical_details(payload)
        return

    render_rag_or_structured_section(
        title="Governed Result",
        route=route,
        status=status,
        result=result if isinstance(result, dict) else None,
        missing_information=missing_information,
        review_required=review_required,
    )
    render_technical_details(payload)


def render_http_error(response: requests.Response) -> None:
    if response.status_code == 422:
        st.error("Please enter a non-empty question before submitting.")
        return

    if response.status_code >= 500:
        render_technical_error(
            "The backend returned a server error. Please try again later.",
            response,
        )
        return

    st.error(f"The backend returned an unexpected HTTP status: {response.status_code}.")


def main() -> None:
    st.set_page_config(page_title="GenAI Sales Assistant", page_icon=":material/bolt:", layout="centered")

    st.title("GenAI Sales Assistant", anchor=False)
    st.write("Ask one product question at a time and inspect the governed result, evidence, and technical details.")
    st.caption(PRIVACY_NOTICE)

    backend_ok = call_health()
    if backend_ok:
        st.caption(f"Backend connected: {API_BASE_URL}")
    else:
        st.warning("Backend unavailable. Start the FastAPI service before submitting a question.")

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    st.markdown("**Example questions**")
    example_columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(example_columns, EXAMPLE_QUESTIONS):
        if column.button(example, use_container_width=True):
            st.session_state.question_input = example

    question = st.text_area(
        "Question",
        key="question_input",
        placeholder="e.g. What is required for PV-surplus charging with ChargeOne 11?",
        height=120,
    )

    if st.button("Submit", type="primary", use_container_width=True):
        if not question.strip():
            st.error("Please enter a non-empty question.")
            return

        with st.spinner("Running governed query..."):
            try:
                response = call_query(question)
            except requests.Timeout:
                st.error("The backend did not respond in time.")
                return
            except requests.RequestException:
                st.error("The backend is currently unavailable.")
                return

        if response.status_code == 200:
            try:
                payload = parse_query_response(response)
            except ValueError:
                render_technical_error(
                    "The backend returned an unexpected response. Please try again later.",
                    response,
                )
                return
            render_payload(payload)
            return

        render_http_error(response)


if __name__ == "__main__":
    main()
