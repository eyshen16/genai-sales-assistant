from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit_app


class StreamlitAppTests(unittest.TestCase):
    def test_parse_query_response_accepts_stable_api_envelope(self) -> None:
        payload = {
            "question": "Question",
            "route": "rag",
            "status": "completed",
            "domains": ["KD-009"],
            "result": {"answer": "Answer"},
            "subresults": [],
            "missing_information": [],
            "review_required": False,
            "router_reason": "rag route",
        }
        response = Mock()
        response.json.return_value = payload

        self.assertEqual(streamlit_app.parse_query_response(response), payload)

    def test_parse_query_response_rejects_malformed_json_or_envelope(self) -> None:
        invalid_json = Mock()
        invalid_json.json.side_effect = ValueError("raw parser detail")
        invalid_shape = Mock()
        invalid_shape.json.return_value = {"status": "completed"}

        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            streamlit_app.parse_query_response(invalid_json)
        with self.assertRaisesRegex(ValueError, "expected API envelope"):
            streamlit_app.parse_query_response(invalid_shape)

    def test_backend_500_shows_generic_message_and_request_reference_only(self) -> None:
        response = Mock(
            status_code=500,
            headers={"X-Request-ID": "request-123"},
            text="secret internal traceback",
        )

        with patch.object(streamlit_app.st, "error") as error, patch.object(
            streamlit_app.st, "caption"
        ) as caption, patch.object(streamlit_app.st, "write") as write:
            streamlit_app.render_http_error(response)

        self.assertIn("server error", error.call_args.args[0].lower())
        self.assertIn("request-123", caption.call_args.args[0])
        write.assert_not_called()
        self.assertNotIn("secret internal traceback", str(error.call_args_list + caption.call_args_list))

    def test_query_call_uses_existing_bounded_frontend_timeout(self) -> None:
        with patch.object(streamlit_app.requests, "post", return_value=Mock()) as post:
            streamlit_app.call_query("Question")

        self.assertEqual(streamlit_app.HTTP_TIMEOUT_SECONDS, 120)
        self.assertEqual(post.call_args.kwargs["timeout"], streamlit_app.HTTP_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
