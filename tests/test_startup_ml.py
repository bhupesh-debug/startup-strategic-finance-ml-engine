"""Unit and integration tests for startup-strategic-finance-ml-engine."""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from startup_finance_ml_pipeline import (
    FIGURES_DIR,
    REPORTS_DIR,
    generate_ai_compute_data,
    generate_ar_collections_data,
    generate_saas_cohort_data,
    run_all,
    run_module_a,
    run_module_b,
    run_module_c,
)


def test_module_a_saas_arr_forecast():
    df = generate_saas_cohort_data(800, seed=101)
    assert len(df) == 800
    assert (df["net_new_arr_m"] > 0).all()
    m, eval_df = run_module_a(df)
    assert len(eval_df) == 200
    assert m["gradient_boosting"]["r2"] > 0.90
    assert m["gradient_boosting"]["mae_m"] < m["baseline_ridge"]["mae_m"]


def test_module_b_ai_compute_margin():
    df = generate_ai_compute_data(800, seed=202)
    m, eval_df = run_module_b(df)
    assert len(eval_df) == 200
    assert m["random_forest"]["r2"] > 0.85
    assert m["random_forest"]["mae_pp"] < m["baseline_ridge"]["mae_pp"]


def test_module_c_ar_collections_risk():
    df = generate_ar_collections_data(1000, seed=303)
    m, payload = run_module_c(df)
    assert len(payload["y_test"]) == 250
    assert m["logistic_regression"]["roc_auc"] > 0.72
    assert m["random_forest"]["roc_auc"] > 0.72


def test_full_pipeline_artifacts():
    summary = run_all()
    assert (REPORTS_DIR / "metrics_summary.json").exists()
    loaded = json.loads((REPORTS_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    assert len(loaded["figures"]) == 4
    for rel in loaded["figures"]:
        p = REPORTS_DIR.parent / rel
        assert p.exists() and p.stat().st_size > 5000
    assert summary["module_a_saas_arr"]["test_rows"] == 300
