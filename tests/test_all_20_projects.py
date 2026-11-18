"""8-Level Verification Test Suite covering All 20 Startup ML Projects & Quant/MC Engine."""

import json
import py_compile
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from all_20_ml_projects import PROJECT_SPECS, run_single_project
from quant_research_and_montecarlo import run_security_audit


def test_level1_static_syntax_and_file_scaffold():
    """Level 1 (Static) & Level 8 (UAT Docs): Every startup project compiles and has README, config, runner."""
    assert len(PROJECT_SPECS) == 20
    for pf in (REPO_ROOT / "src").glob("*.py"):
        py_compile.compile(str(pf), doraise=True)
    for spec in PROJECT_SPECS:
        p_dir = REPO_ROOT / "projects" / spec["slug"]
        assert (p_dir / "README.md").exists()
        assert (p_dir / "config.json").exists()
        assert (p_dir / "run_project.py").exists()
        py_compile.compile(str(p_dir / "run_project.py"), doraise=True)


def test_level2_unit_all_20_startup_models():
    """Level 2 (Unit) & Level 7 (Performance): Run every one of the 20 models cleanly with quality & speed gates."""
    for spec in PROJECT_SPECS:
        res = run_single_project(spec)
        assert res["ml_primary"] >= 0.78, f"Low score for {spec['slug']}: {res}"
        assert res["elapsed_ms"] < 4000.0, f"Slow runtime for {spec['slug']}: {res['elapsed_ms']}ms"
        if spec["task_type"] == "regression":
            assert res["ml_primary"] > res["baseline_primary"], f"Expected ML > Baseline for {spec['slug']}"


def test_level3_and_4_e2e_cli_and_quant_outputs():
    """Level 3 (Integration) & Level 4 (E2E): Standalone runners & quant/MC JSON artifacts verified."""
    for slug in ["01_glide_ai_billing_forecast", "15_captivateiq_gtm_quota_commissions", "17_openai_workloads_inference_frontier", "20_alphasense_o2c_ar_delinquency"]:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "projects" / slug / "run_project.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        assert payload["slug"] == slug
        assert payload["ml_primary"] >= 0.80

    q_path = REPO_ROOT / "reports" / "repo_quant_and_mc_summary.json"
    assert q_path.exists()
    q_data = json.loads(q_path.read_text(encoding="utf-8"))
    assert len(q_data["hypotheses"]) == 5
    assert q_data["monte_carlo"]["trials"] == 10000
    scen = q_data["sensitivity_and_scenarios"]["scenarios"]
    assert scen["worst_case"]["probability_pct"] + scen["base_case"]["probability_pct"] + scen["best_case"]["probability_pct"] == 100


def test_level5_and_6_regression_and_security():
    """Level 5 (Regression) & Level 6 (Security): Zero hardcoded secrets, zero unsafe eval/exec/shell."""
    sec = run_security_audit()
    assert sec["p0_critical_secrets"] == 0
    assert sec["unsafe_exec_or_shell"] == 0
    assert sec["status"].startswith("PASS")
    for idx in range(5, 9):
        figs = list((REPO_ROOT / "reports" / "figures").glob(f"fig{idx}_*.png"))
        assert figs and figs[0].stat().st_size > 5000
