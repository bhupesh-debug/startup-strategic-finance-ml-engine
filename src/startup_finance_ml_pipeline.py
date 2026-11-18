"""
Startup Strategic Finance, AI Compute & O2C ML Engine
-----------------------------------------------------
Unified 3-Module Applied ML Pipeline mapped across Top 20 Prioritized Startups:
  - Module A: SaaS Net-New ARR & Burn Multiple Forecasting (CaptivateIQ, Vendelux, Narvar, Grafana, Glide)
  - Module B: AI Workload GPU Compute Margin & Latency Optimizer (OpenAI Workloads, Baseten, Juniper Square)
  - Module C: Order-to-Cash (O2C) 60+ Day AR Delinquency Risk Scorer (AlphaSense, Hera, Glide, FORT)

All training/validation splits are strictly leakage-free (separate holdout test sets).
Synthetic benchmarking data is explicitly labeled [SYN] in reports and schemas.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# MODULE A: SaaS Net-New ARR & Burn Multiple Forecasting [SYN]
# ---------------------------------------------------------------------------
def generate_saas_cohort_data(n: int = 1200, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    starting_arr_m = rng.uniform(5.0, 180.0, n)
    ramping_aes = rng.integers(4, 95, n)
    quota_attainment = rng.uniform(0.58, 1.18, n)
    nrr = rng.uniform(0.96, 1.32, n)
    pipeline_coverage = rng.uniform(1.8, 5.2, n)
    cac_payback_mo = rng.uniform(7.0, 28.0, n)
    sm_spend_m = starting_arr_m * rng.uniform(0.22, 0.58, n) / 4.0

    # Non-linear ARR & Burn Multiple ground truth with realistic interactions
    gross_new_arr_m = (
        0.18 * ramping_aes * (quota_attainment ** 1.25) * np.sqrt(pipeline_coverage / 3.0)
        + 0.045 * starting_arr_m * (nrr - 1.0) * 4.0
        + rng.normal(0, 0.45, n)
    )
    net_new_arr_m = np.maximum(0.25, gross_new_arr_m)
    net_burn_m = np.maximum(
        0.20,
        sm_spend_m * 0.92 + 0.08 * starting_arr_m / 4.0 - 0.14 * net_new_arr_m + rng.normal(0, 0.35, n),
    )
    burn_multiple = np.round(net_burn_m / np.maximum(0.4, net_new_arr_m), 3)

    return pd.DataFrame(
        {
            "starting_arr_m": np.round(starting_arr_m, 2),
            "ramping_aes": ramping_aes,
            "quota_attainment": np.round(quota_attainment, 3),
            "nrr": np.round(nrr, 3),
            "pipeline_coverage": np.round(pipeline_coverage, 2),
            "cac_payback_mo": np.round(cac_payback_mo, 1),
            "sm_spend_m": np.round(sm_spend_m, 2),
            "net_burn_m": np.round(net_burn_m, 2),
            "net_new_arr_m": np.round(net_new_arr_m, 3),
            "burn_multiple": burn_multiple,
        }
    )


def run_module_a(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    features = [
        "starting_arr_m",
        "ramping_aes",
        "quota_attainment",
        "nrr",
        "pipeline_coverage",
        "cac_payback_mo",
        "sm_spend_m",
    ]
    X = df[features]
    y = df["net_new_arr_m"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    baseline = Ridge(alpha=1.0).fit(X_train_sc, y_train)
    pred_base = baseline.predict(X_test_sc)

    gbr = GradientBoostingRegressor(
        n_estimators=180, max_depth=3, learning_rate=0.06, random_state=42
    ).fit(X_train, y_train)
    pred_gbr = gbr.predict(X_test)

    eval_df = X_test.copy()
    eval_df["actual_net_new_arr_m"] = y_test.values
    eval_df["pred_baseline_m"] = pred_base
    eval_df["pred_gbr_m"] = pred_gbr
    eval_df["burn_multiple"] = df.loc[X_test.index, "burn_multiple"].values

    metrics = {
        "rows_total": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_ridge": {
            "r2": round(float(r2_score(y_test, pred_base)), 4),
            "mae_m": round(float(mean_absolute_error(y_test, pred_base)), 4),
            "rmse_m": round(float(root_mean_squared_error(y_test, pred_base)), 4),
        },
        "gradient_boosting": {
            "r2": round(float(r2_score(y_test, pred_gbr)), 4),
            "mae_m": round(float(mean_absolute_error(y_test, pred_gbr)), 4),
            "rmse_m": round(float(root_mean_squared_error(y_test, pred_gbr)), 4),
        },
        "top_feature_importances": {
            f: round(float(imp), 4)
            for f, imp in sorted(zip(features, gbr.feature_importances_), key=lambda x: -x[1])
        },
    }
    return metrics, eval_df


# ---------------------------------------------------------------------------
# MODULE B: AI Workload Inference & GPU Compute Margin Optimizer [SYN]
# ---------------------------------------------------------------------------
def generate_ai_compute_data(n: int = 1200, seed: int = 202) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tokens_per_min_k = rng.uniform(80.0, 2400.0, n)
    gpu_utilization = rng.uniform(0.28, 0.94, n)
    reserved_gpu_share = rng.uniform(0.20, 0.90, n)
    batch_size = rng.choice([8, 16, 32, 64], n, p=[0.20, 0.35, 0.30, 0.15])
    realized_price_per_m_tok = rng.uniform(1.80, 6.50, n)

    p99_latency_ms = np.round(
        115.0 + 1.85 * batch_size + 42.0 * np.exp(2.1 * (gpu_utilization - 0.65)) + rng.normal(0, 9.0, n),
        1,
    )
    unit_compute_cost_per_m_tok = np.round(
        (0.95 / np.maximum(0.28, gpu_utilization))
        * (1.0 - 0.24 * reserved_gpu_share)
        * (1.0 + 0.002 * (batch_size == 8).astype(float) * 60.0)
        + rng.normal(0, 0.09, n),
        3,
    )
    unit_compute_cost_per_m_tok = np.maximum(0.25, unit_compute_cost_per_m_tok)
    gross_margin_pct = np.round(
        (realized_price_per_m_tok - unit_compute_cost_per_m_tok) / realized_price_per_m_tok * 100.0,
        2,
    )
    return pd.DataFrame(
        {
            "tokens_per_min_k": np.round(tokens_per_min_k, 1),
            "gpu_utilization": np.round(gpu_utilization, 3),
            "reserved_gpu_share": np.round(reserved_gpu_share, 3),
            "batch_size": batch_size,
            "realized_price_per_m_tok": np.round(realized_price_per_m_tok, 2),
            "p99_latency_ms": p99_latency_ms,
            "unit_compute_cost_per_m_tok": unit_compute_cost_per_m_tok,
            "gross_margin_pct": gross_margin_pct,
        }
    )


def run_module_b(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    features = [
        "tokens_per_min_k",
        "gpu_utilization",
        "reserved_gpu_share",
        "batch_size",
        "realized_price_per_m_tok",
        "p99_latency_ms",
    ]
    X = df[features]
    y = df["gross_margin_pct"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    base = Ridge(alpha=1.0).fit(scaler.fit_transform(X_train), y_train)
    pred_base = base.predict(scaler.transform(X_test))

    rf = RandomForestRegressor(n_estimators=160, max_depth=10, random_state=42).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    eval_df = X_test.copy()
    eval_df["actual_margin_pct"] = y_test.values
    eval_df["pred_rf_margin_pct"] = pred_rf
    eval_df["residual_margin_pp"] = eval_df["actual_margin_pct"] - eval_df["pred_rf_margin_pct"]

    # Anomaly threshold: residual < -1.96 * std OR margin < 20% with low utilization (< 0.45)
    z_std = float(eval_df["residual_margin_pp"].std())
    anomalies = eval_df[
        (eval_df["residual_margin_pp"] < -1.96 * z_std)
        | ((eval_df["actual_margin_pct"] < 20.0) & (eval_df["gpu_utilization"] < 0.45))
    ]

    metrics = {
        "rows_total": int(len(df)),
        "test_rows": int(len(X_test)),
        "baseline_ridge": {
            "r2": round(float(r2_score(y_test, pred_base)), 4),
            "mae_pp": round(float(mean_absolute_error(y_test, pred_base)), 3),
        },
        "random_forest": {
            "r2": round(float(r2_score(y_test, pred_rf)), 4),
            "mae_pp": round(float(mean_absolute_error(y_test, pred_rf)), 3),
            "rmse_pp": round(float(root_mean_squared_error(y_test, pred_rf)), 3),
        },
        "margin_anomaly_clusters_flagged": int(len(anomalies)),
        "mean_gross_margin_pct": round(float(df["gross_margin_pct"].mean()), 2),
        "mean_p99_latency_ms": round(float(df["p99_latency_ms"].mean()), 1),
    }
    return metrics, eval_df


# ---------------------------------------------------------------------------
# MODULE C: Order-to-Cash & 60+ Day AR Delinquency Risk Scorer [SYN]
# ---------------------------------------------------------------------------
def generate_ar_collections_data(n: int = 1500, seed: int = 303) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    invoice_amt_k = rng.lognormal(mean=3.2, sigma=0.75, size=n)
    days_to_renewal = rng.integers(-30, 365, n)
    prior_late_count = rng.poisson(1.15, n)
    open_billing_disputes = rng.binomial(2, 0.16, n)
    avg_hist_pay_days = rng.normal(34.0, 14.0, n)
    auto_pay_enabled = rng.binomial(1, 0.38, n)

    logit = (
        -2.25
        + 0.72 * prior_late_count
        + 1.35 * open_billing_disputes
        + 0.048 * np.maximum(0.0, avg_hist_pay_days - 30.0)
        - 1.10 * auto_pay_enabled
        + 0.35 * (days_to_renewal < 30).astype(float)
        + 0.008 * np.minimum(120.0, invoice_amt_k)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    delinquent_60d = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "invoice_amt_k": np.round(invoice_amt_k, 2),
            "days_to_renewal": days_to_renewal,
            "prior_late_count": prior_late_count,
            "open_billing_disputes": open_billing_disputes,
            "avg_hist_pay_days": np.round(np.maximum(5.0, avg_hist_pay_days), 1),
            "auto_pay_enabled": auto_pay_enabled,
            "delinquent_60d": delinquent_60d,
        }
    )


def run_module_c(df: pd.DataFrame) -> tuple[dict, dict]:
    features = [
        "invoice_amt_k",
        "days_to_renewal",
        "prior_late_count",
        "open_billing_disputes",
        "avg_hist_pay_days",
        "auto_pay_enabled",
    ]
    X = df[features]
    y = df["delinquent_60d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    logreg = LogisticRegression(max_iter=500, random_state=42).fit(scaler.fit_transform(X_train), y_train)
    p_log = logreg.predict_proba(scaler.transform(X_test))[:, 1]

    rf = RandomForestClassifier(
        n_estimators=180, max_depth=7, min_samples_leaf=8, random_state=42
    ).fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]

    ar_at_risk_m = float(
        df.loc[X_test.index, "invoice_amt_k"][p_rf >= 0.50].sum() / 1000.0
    )

    metrics = {
        "rows_total": int(len(df)),
        "test_rows": int(len(X_test)),
        "delinquency_base_rate_pct": round(float(y.mean() * 100.0), 2),
        "logistic_regression": {
            "roc_auc": round(float(roc_auc_score(y_test, p_log)), 4),
            "pr_auc": round(float(average_precision_score(y_test, p_log)), 4),
            "brier": round(float(brier_score_loss(y_test, p_log)), 4),
        },
        "random_forest": {
            "roc_auc": round(float(roc_auc_score(y_test, p_rf)), 4),
            "pr_auc": round(float(average_precision_score(y_test, p_rf)), 4),
            "brier": round(float(brier_score_loss(y_test, p_rf)), 4),
        },
        "flagged_high_risk_ar_volume_m": round(ar_at_risk_m, 3),
    }
    plot_payload = {"y_test": y_test.values, "p_log": p_log, "p_rf": p_rf}
    return metrics, plot_payload


# ---------------------------------------------------------------------------
# FIGURES (Publication-Grade 150 DPI Matplotlib Outputs)
# ---------------------------------------------------------------------------
def build_figures(
    eval_a: pd.DataFrame,
    metrics_a: dict,
    eval_b: pd.DataFrame,
    metrics_b: dict,
    payload_c: dict,
    metrics_c: dict,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#CBD5E1"})

    # Fig 1: SaaS Net-New ARR Actual vs Predicted + Burn Multiple
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    ax0 = axes[0]
    ax0.scatter(eval_a["actual_net_new_arr_m"], eval_a["pred_gbr_m"], alpha=0.55, color="#1E3A5F", s=22, label="GBR Pred")
    lim_max = float(max(eval_a["actual_net_new_arr_m"].max(), eval_a["pred_gbr_m"].max()) * 1.05)
    ax0.plot([0, lim_max], [0, lim_max], "r--", lw=1.4, label="Ideal Fit (y=x)")
    ax0.set_title(f"Module A: Net-New ARR Forecast (R²={metrics_a['gradient_boosting']['r2']:.3f}, MAE=${metrics_a['gradient_boosting']['mae_m']:.2f}M)")
    ax0.set_xlabel("Actual Quarterly Net-New ARR ($M) [SYN]")
    ax0.set_ylabel("Predicted Net-New ARR ($M)")
    ax0.legend(frameon=True)
    ax0.grid(alpha=0.25)

    ax1 = axes[1]
    imp_items = list(metrics_a["top_feature_importances"].items())[:6]
    names = [k.replace("_", " ").title() for k, _ in reversed(imp_items)]
    vals = [v * 100 for _, v in reversed(imp_items)]
    ax1.barh(names, vals, color="#006D77")
    ax1.set_title("SaaS ARR Forecast Feature Importance (%)")
    ax1.set_xlabel("Relative Feature Weight (%)")
    ax1.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_saas_arr_burn_multiple_forecast.png")
    plt.close(fig)

    # Fig 2: AI GPU Compute Margin vs Utilization & Latency Tradeoff
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    sc = axes[0].scatter(
        eval_b["gpu_utilization"] * 100,
        eval_b["actual_margin_pct"],
        c=eval_b["p99_latency_ms"],
        cmap="viridis",
        alpha=0.65,
        s=24,
    )
    axes[0].axhline(65.0, color="#C1121F", ls="--", lw=1.2, label="Target Gross Margin Floor (65%)")
    axes[0].set_title(f"Module B: GPU Utilization vs Gross Margin (RF R²={metrics_b['random_forest']['r2']:.3f})")
    axes[0].set_xlabel("GPU Cluster Utilization (%) [SYN]")
    axes[0].set_ylabel("Inference Gross Margin (%)")
    axes[0].legend(frameon=True, loc="lower right")
    axes[0].grid(alpha=0.25)
    plt.colorbar(sc, ax=axes[0], label="P99 Latency (ms)")

    axes[1].scatter(eval_b["p99_latency_ms"], eval_b["actual_margin_pct"], color="#5A189A", alpha=0.5, s=20)
    axes[1].set_title(f"Latency vs Compute Margin ({metrics_b['margin_anomaly_clusters_flagged']} Anomalies Flagged)")
    axes[1].set_xlabel("P99 Serving Latency (ms)")
    axes[1].set_ylabel("Inference Gross Margin (%)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_ai_gpu_compute_margin_curve.png")
    plt.close(fig)

    # Fig 3: AR Collections ROC & Calibration Curve
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fpr_l, tpr_l, _ = roc_curve(payload_c["y_test"], payload_c["p_log"])
    fpr_r, tpr_r, _ = roc_curve(payload_c["y_test"], payload_c["p_rf"])
    axes[0].plot(fpr_l, tpr_l, label=f"LogReg Baseline (AUC={metrics_c['logistic_regression']['roc_auc']:.3f})", color="#E07B39", lw=2)
    axes[0].plot(fpr_r, tpr_r, label=f"RandomForest (AUC={metrics_c['random_forest']['roc_auc']:.3f})", color="#2D6A4F", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("Module C: 60+ Day AR Delinquency ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(frameon=True)
    axes[0].grid(alpha=0.25)

    frac_pos, mean_pred = calibration_curve(payload_c["y_test"], payload_c["p_log"], n_bins=8)
    frac_rf, mean_rf = calibration_curve(payload_c["y_test"], payload_c["p_rf"], n_bins=8)
    axes[1].plot(mean_pred, frac_pos, "s-", color="#E07B39", label=f"LogReg (Brier={metrics_c['logistic_regression']['brier']:.3f})")
    axes[1].plot(mean_rf, frac_rf, "o-", color="#2D6A4F", label=f"RF (Brier={metrics_c['random_forest']['brier']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("Collections Risk Probability Calibration")
    axes[1].set_xlabel("Mean Predicted Delinquency Risk")
    axes[1].set_ylabel("Observed Delinquency Fraction")
    axes[1].legend(frameon=True)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_ar_collections_roc_calibration.png")
    plt.close(fig)

    # Fig 4: Startup Unit Economics Waterfall across 4 Quota/NRR Regimes
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=150)
    cohorts = ["Seed <$10M\nARR", "Series A/B\n($10M-$40M)", "Growth C\n($40M-$100M)", "Late-Stage\n($100M+)"]
    bins = pd.cut(eval_a["starting_arr_m"], bins=[0, 10, 40, 100, 200], labels=cohorts)
    grp = eval_a.groupby(bins, observed=False).agg(
        avg_pred_arr=("pred_gbr_m", "mean"),
        avg_burn_mult=("burn_multiple", "mean"),
    ).fillna(0)
    x = np.arange(len(cohorts))
    w = 0.36
    b1 = ax.bar(x - w / 2, grp["avg_pred_arr"], width=w, color="#1E3A5F", label="Avg Pred Net-New ARR ($M/Qtr)")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, grp["avg_burn_mult"], width=w, color="#E07B39", label="Avg Burn Multiple (x)")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Predicted Net-New ARR ($M)")
    ax2.set_ylabel("Burn Multiple (Net Burn / Net-New ARR)")
    ax.set_title("Cross-Stage SaaS Capital Efficiency: Predicted Net-New ARR vs. Burn Multiple [SYN]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_startup_cohort_unit_economics_waterfall.png")
    plt.close(fig)


def run_all() -> dict:
    ensure_dirs()
    df_a = generate_saas_cohort_data()
    df_b = generate_ai_compute_data()
    df_c = generate_ar_collections_data()

    df_a.head(200).to_csv(DATA_DIR / "sample_saas_cohorts_syn.csv", index=False)
    df_b.head(200).to_csv(DATA_DIR / "sample_ai_compute_syn.csv", index=False)
    df_c.head(200).to_csv(DATA_DIR / "sample_ar_collections_syn.csv", index=False)

    metrics_a, eval_a = run_module_a(df_a)
    metrics_b, eval_b = run_module_b(df_b)
    metrics_c, payload_c = run_module_c(df_c)

    build_figures(eval_a, metrics_a, eval_b, metrics_b, payload_c, metrics_c)

    summary = {
        "data_tag": "[SYN] Deterministic Synthetic Startup Portfolio Benchmarks",
        "module_a_saas_arr": metrics_a,
        "module_b_ai_compute_margin": metrics_b,
        "module_c_ar_collections_risk": metrics_c,
        "figures": [
            "reports/figures/fig1_saas_arr_burn_multiple_forecast.png",
            "reports/figures/fig2_ai_gpu_compute_margin_curve.png",
            "reports/figures/fig3_ar_collections_roc_calibration.png",
            "reports/figures/fig4_startup_cohort_unit_economics_waterfall.png",
        ],
    }
    (REPORTS_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_all()
    print(json.dumps(res, indent=2))
