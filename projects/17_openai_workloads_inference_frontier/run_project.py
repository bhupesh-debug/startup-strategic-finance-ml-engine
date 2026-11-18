"""Standalone CLI runner for #17 OpenAI — Inference P99 Latency vs GPU Utilization Cost Efficiency Optimizer."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from all_20_ml_projects import PROJECT_SPECS, run_single_project

if __name__ == "__main__":
    spec = next(s for s in PROJECT_SPECS if s["slug"] == "17_openai_workloads_inference_frontier")
    print(json.dumps(run_single_project(spec), indent=2))
