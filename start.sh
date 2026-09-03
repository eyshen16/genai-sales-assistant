#!/usr/bin/env bash

set -euo pipefail

uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 &
api_pid=$!

streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" &
streamlit_pid=$!

cleanup() {
  kill "${api_pid}" "${streamlit_pid}" 2>/dev/null || true
}

trap cleanup TERM INT EXIT

wait -n "${api_pid}" "${streamlit_pid}"
exit_code=$?

kill "${api_pid}" "${streamlit_pid}" 2>/dev/null || true
wait "${api_pid}" 2>/dev/null || true
wait "${streamlit_pid}" 2>/dev/null || true

exit "${exit_code}"
