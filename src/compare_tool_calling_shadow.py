"""Run the frozen Tool Calling II coexistence evaluation."""
from __future__ import annotations

import argparse
import json

from tool_calling_shadow import run_shadow_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run three model attempts for every gated case.")
    parser.add_argument("--output", help="Append-safe JSONL result path; the file is reset at run start.")
    args = parser.parse_args()
    print(json.dumps(run_shadow_evaluation(live=args.live, output_path=args.output), indent=2))
