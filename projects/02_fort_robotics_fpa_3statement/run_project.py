"""Standalone CLI runner for #02 FORT Robotics — Hardware/Software/Services Blended Margin & 13-Wk Cash Model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from all_20_ml_projects import PROJECT_SPECS, run_single_project

if __name__ == "__main__":
    spec = next(s for s in PROJECT_SPECS if s["slug"] == "02_fort_robotics_fpa_3statement")
    print(json.dumps(run_single_project(spec), indent=2))
