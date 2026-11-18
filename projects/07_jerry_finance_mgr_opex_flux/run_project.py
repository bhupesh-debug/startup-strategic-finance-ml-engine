"""Standalone CLI runner for #07 Jerry.ai — Marketplace OpEx Variance Flux & Vendor Spend Anomaly Classifier."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from all_20_ml_projects import PROJECT_SPECS, run_single_project

if __name__ == "__main__":
    spec = next(s for s in PROJECT_SPECS if s["slug"] == "07_jerry_finance_mgr_opex_flux")
    print(json.dumps(run_single_project(spec), indent=2))
