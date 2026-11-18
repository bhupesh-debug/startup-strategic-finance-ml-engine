"""
Unified 20-Project ML Runner & Generator for All 20 Prioritized Startups.
Implements leakage-free regression and classification pipelines across 20 startup finance,
AI compute, SaaS GTM, healthcare marketplace, and technical accounting/O2C domains [SYN].
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"
REPORTS_DIR = REPO_ROOT / "reports"

PROJECT_SPECS: list[dict[str, Any]] = [
    {
        "rank": 1,
        "slug": "01_glide_ai_billing_forecast",
        "company": "Glide",
        "role": "Strategic Finance Lead",
        "project_title": "AI Contract-to-Cash Billing & Cash Runway Forecaster",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1001,
        "n_rows": 900,
        "target": "net_cash_collected_k",
        "unit": "$K",
        "features": ["fintech_contracts", "usage_overage_pct", "auto_recon_rate", "days_to_invoice", "cogs_alloc_ratio", "churn_rate"],
        "domain_context": "Community-bank embedded fintech contract billing, AI auto-reconciliation, and monthly runway cash collection ($140K-$180K + eq NYC).",
        "talking_point": "Models nonlinear impact of contract-to-cash invoice lag and usage overages on monthly cash runway with automated SQL billing tie-out."
    },
    {
        "rank": 2,
        "slug": "02_fort_robotics_fpa_3statement",
        "company": "FORT Robotics",
        "role": "FP&A Manager",
        "project_title": "Hardware/Software/Services Blended Margin & 13-Wk Cash Model",
        "task_type": "regression",
        "model_family": "RF",
        "seed": 1002,
        "n_rows": 900,
        "target": "operating_cash_flow_k",
        "unit": "$K",
        "features": ["hardware_units", "sw_attach_pct", "bom_cost_index", "inventory_days", "deferred_rev_ratio", "headcount_ramp"],
        "domain_context": "Public-readiness / de-SPAC 3-statement driver model bridging robotics hardware inventory turns and ratable software ARR.",
        "talking_point": "Prevents working-capital cash traps by tying BOM inventory lead times directly into the 13-week cash bridge alongside ASC 606 mix."
    },
    {
        "rank": 3,
        "slug": "03_covera_health_care_capacity",
        "company": "Covera Health",
        "role": "Manager, FP&A",
        "project_title": "Care Coordination Scan Volume & FTE Staffing Capacity Optimizer",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1003,
        "n_rows": 900,
        "target": "cost_per_scan_usd",
        "unit": "$",
        "features": ["weekly_scan_orders", "fte_utilization", "agent_ramp_share", "attrition_pct", "ai_triage_adoption", "offshore_mix"],
        "domain_context": "Workforce staffing and cost-per-order unit economics for clinical Care Coordination ops (NYC optional).",
        "talking_point": "Quantifies cost-per-scan elasticity under seasonal demand spikes, agent attrition cohorts, and AI triage routing."
    },
    {
        "rank": 4,
        "slug": "04_narvar_saas_transactional_arr",
        "company": "Narvar",
        "role": "Senior Manager, FP&A",
        "project_title": "B2B SaaS + Transactional Shipping Volume & NRR Revenue Engine",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1004,
        "n_rows": 1000,
        "target": "quarterly_net_new_rev_k",
        "unit": "$K",
        "features": ["base_saas_arr_m", "holiday_ship_mult", "nrr_rate", "take_rate_bps", "ramping_aes", "quota_attain"],
        "domain_context": "Post-purchase SaaS + peak Q4 transactional parcel volume forecasting and AOP/LRP modeling ($157K-$225K Remote).",
        "talking_point": "Decouples deterministic base SaaS NRR from volatile merchant shipping GMV elasticity so BvA variances stay transparent."
    },
    {
        "rank": 5,
        "slug": "05_jerry_strategic_finance_ltv",
        "company": "Jerry.ai",
        "role": "Strategic Finance Manager",
        "project_title": "InsurTech Vertical CAC Payback, Policy Retention & LTV Model",
        "task_type": "regression",
        "model_family": "RF",
        "seed": 1005,
        "n_rows": 1000,
        "target": "cohort_24m_ltv_usd",
        "unit": "$",
        "features": ["policy_premium_usd", "carrier_take_rate", "month6_retention", "cac_usd", "claim_assist_score", "multi_product_attach"],
        "domain_context": "AI car-ownership & insurance marketplace underwriting across new verticals and paid channels ($150K-$225K Remote).",
        "talking_point": "Maps Bankers Life insurance sensitivity analytics onto consumer policy persistence, carrier commission mix, and Month-12 payback gates."
    },
    {
        "rank": 6,
        "slug": "06_baseten_gpu_compute_margin",
        "company": "Baseten",
        "role": "Strategic Finance",
        "project_title": "AI Inference GPU Cluster Utilization & Gross Margin Optimizer",
        "task_type": "regression",
        "model_family": "RF",
        "seed": 1006,
        "n_rows": 1000,
        "target": "inference_gross_margin_pct",
        "unit": "%",
        "features": ["gpu_utilization", "reserved_share", "tokens_k_per_min", "realized_price_m_tok", "batch_size", "p99_latency_ms"],
        "domain_context": "Usage-based AI inference revenue, GPU utilization efficiency, enterprise deal minimums & margin analysis ($200K-$280K NYC).",
        "talking_point": "Pinpoints exact GPU utilization thresholds and spot/reserved cluster splits needed to protect >=65% inference gross margins."
    },
    {
        "rank": 7,
        "slug": "07_jerry_finance_mgr_opex_flux",
        "company": "Jerry.ai",
        "role": "Finance Manager",
        "project_title": "Marketplace OpEx Variance Flux & Vendor Spend Anomaly Classifier",
        "task_type": "classification",
        "model_family": "RF_CLF",
        "seed": 1007,
        "n_rows": 1000,
        "target": "unbudgeted_overrun_flag",
        "unit": "prob",
        "features": ["vendor_spend_ratio", "po_missing_flag", "dup_seat_pct", "dept_headcount_delta", "contract_auto_renew", "prior_flux_breaches"],
        "domain_context": "Operational FP&A close control, Ramp/NetSuite spend underwriting, and monthly vendor flux escalation ($150K-$225K Remote).",
        "talking_point": "Flags duplicate SaaS seats, missing POs, and auto-renewal traps before month-end close locks OpEx overspend."
    },
    {
        "rank": 8,
        "slug": "08_juniper_square_rd_capital",
        "company": "Juniper Square",
        "role": "Finance Director, R&D",
        "project_title": "R&D Engineering Stage-Gate ROI & ASC 350-40 Capitalization Model",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1008,
        "n_rows": 900,
        "target": "product_net_roi_pct",
        "unit": "%",
        "features": ["eng_pod_cost_k", "india_us_blend_ratio", "cloud_ai_compute_k", "asc350_cap_pct", "arr_unlock_k", "sprint_velocity_idx"],
        "domain_context": "CTO/CPO Finance partnering across US & India R&D hubs, Pigment/TeamOhana headcount, cloud AI cost & cap-vs-expense ($175K-$220K).",
        "talking_point": "Aligns Engineering squad velocity and blended US/India cost efficiency with auditable ASC 350-40 software capitalization gates."
    },
    {
        "rank": 9,
        "slug": "09_zocdoc_ir_booking_take_rate",
        "company": "Zocdoc",
        "role": "Senior Manager, Strategic Finance & IR",
        "project_title": "Patient Marketplace Booking Volume, Take-Rate & IR Bridge Engine",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1009,
        "n_rows": 900,
        "target": "net_marketplace_rev_k",
        "unit": "$K",
        "features": ["patient_bookings_k", "enterprise_system_mix", "effective_take_rate_usd", "provider_churn_pct", "rebook_share", "cac_efficiency"],
        "domain_context": "NYC Hybrid Strategic Finance & IR bridge decomposing volume beats vs. enterprise health-system take-rate mix.",
        "talking_point": "Produces investor-grade volume/mix/rate bridges explaining why lower enterprise take-rate expands multi-year LTV."
    },
    {
        "rank": 10,
        "slug": "10_vendelux_crm_maxio_qbo_tieout",
        "company": "Vendelux",
        "role": "FP&A Manager",
        "project_title": "HubSpot-Maxio-QBO Three-Way Reconciliation & Series C Audit Scorer",
        "task_type": "classification",
        "model_family": "RF_CLF",
        "seed": 1010,
        "n_rows": 1000,
        "target": "recon_break_flag",
        "unit": "prob",
        "features": ["crm_maxio_amt_diff_pct", "start_date_lag_days", "custom_ramp_flag", "qbo_deferred_mismatch", "manual_adj_count", "multi_year_contract"],
        "domain_context": "Series B NYC SaaS FP&A unifying HubSpot bookings, Maxio subscriptions & QuickBooks Online GAAP ledger ($140K-$200K + eq).",
        "talking_point": "Eliminates fundraising data-room breaks by catching CRM-to-Maxio start-date lags and custom ramp schedules before close."
    },
    {
        "rank": 11,
        "slug": "11_maven_clinic_client_rev_bva",
        "company": "Maven Clinic",
        "role": "Strategic Finance Associate",
        "project_title": "Employer Client Lives, Product Mix & Monthly BvA Forecast Model",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1011,
        "n_rows": 900,
        "target": "client_monthly_rev_k",
        "unit": "$K",
        "features": ["enrolled_lives_k", "pmpm_fee_usd", "utilization_tier", "enterprise_product_count", "clinical_margin_pct", "impl_lag_days"],
        "domain_context": "NYC Hybrid quantitative client/product revenue forecasting and Accounting close partnering ($94K-$110K + eq).",
        "talking_point": "Links employer launch dates and product module attach directly to monthly GAAP BvA reviews."
    },
    {
        "rank": 12,
        "slug": "12_fort_robotics_sec_sox_controls",
        "company": "FORT Robotics",
        "role": "Head of SEC Reporting",
        "project_title": "ASC 606 Multi-Element SSP & SOX 302/404 Control Exception Scorer",
        "task_type": "classification",
        "model_family": "RF_CLF",
        "seed": 1012,
        "n_rows": 1000,
        "target": "sox_control_exception_flag",
        "unit": "prob",
        "features": ["ssp_deviation_pct", "manual_je_late_flag", "sod_conflict_score", "multi_elem_bundle_cnt", "subledger_tieout_diff_k", "cut_off_days"],
        "domain_context": "Public-company external reporting readiness, SOX 302/404 audit controls & Workiva XBRL close validation ($140K-$220K).",
        "talking_point": "Demonstrates automated pre-audit detection of Segregation-of-Duties (SoD) and SSP allocation risk (framed transparently under CPA-candidate track)."
    },
    {
        "rank": 13,
        "slug": "13_grafana_labs_corp_fpa_lrp",
        "company": "Grafana Labs",
        "role": "Senior Director, Corporate FP&A",
        "project_title": "Consolidated 3-Statement LRP & Rule-of-40 Public-Readiness Engine",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1013,
        "n_rows": 1000,
        "target": "rule_of_40_score",
        "unit": "pts",
        "features": ["arr_yoy_growth_pct", "fcf_margin_pct", "ndr_pct", "sm_pct_rev", "rd_pct_rev", "tax_eff_rate"],
        "domain_context": "Enterprise SaaS Corporate FP&A consolidated P&L/BS/CF forecasting in Pigment and close-to-insight rhythm ($268K-$322K OTE).",
        "talking_point": "Quantifies trade-offs between top-line ARR growth investment and FCF margin levers across long-range plans."
    },
    {
        "rank": 14,
        "slug": "14_the_farmers_dog_cohort_irr",
        "company": "The Farmer's Dog",
        "role": "Director, Finance",
        "project_title": "Subscription vs Retail Multi-Channel Cohort & IRR Guardrail Model",
        "task_type": "regression",
        "model_family": "RF",
        "seed": 1014,
        "n_rows": 1000,
        "target": "channel_18m_irr_pct",
        "unit": "%",
        "features": ["promo_discount_pct", "spoilage_shipping_pct", "sub_month6_retain", "retail_halo_lift", "cac_payback_mo", "aov_usd"],
        "domain_context": "NYC SoHo Director of Finance modeling multi-product/multi-channel promo guardrails, cohort payback, and IRR ($178K-$200K).",
        "talking_point": "Separates true incremental multi-channel IRR from promo-discounted low-LTV acquisition cohorts."
    },
    {
        "rank": 15,
        "slug": "15_captivateiq_gtm_quota_commissions",
        "company": "CaptivateIQ",
        "role": "Strategic Finance Manager, GTM",
        "project_title": "GTM Sales Capacity, AE Ramp & Commission Accelerator Forecaster",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1015,
        "n_rows": 1000,
        "target": "net_new_arr_m",
        "unit": "$M",
        "features": ["ramping_aes", "quota_attainment", "pipeline_coverage", "accelerator_mult", "nrr", "sales_cycle_days"],
        "domain_context": "E-Verify Tier-1 Remote GTM Strategic Finance owning capacity planning, net new ARR, CaptivateIQ commissions & Adaptive.",
        "talking_point": "Links AE ramp tenure curves and Salesforce pipeline coverage directly into net new ARR and marginal commission cost-of-ARR."
    },
    {
        "rank": 16,
        "slug": "16_netlify_plg_cloud_margin",
        "company": "Netlify [Archived Template]",
        "role": "Senior Strategic Finance Manager",
        "project_title": "PLG Free-to-Enterprise Conversion & Hosting Bandwidth Margin Classifier",
        "task_type": "classification",
        "model_family": "RF_CLF",
        "seed": 1016,
        "n_rows": 900,
        "target": "enterprise_upgrade_flag",
        "unit": "prob",
        "features": ["team_seats_active", "monthly_bandwidth_gb", "ci_build_mins", "sso_security_hits", "days_since_signup", "billing_card_flag"],
        "domain_context": "Archived Greenhouse template preserved for Developer Cloud PLG-to-Enterprise PQL conversion & compute margin caps.",
        "talking_point": "Identifies PQL signals where free compute consumption justifies proactive Enterprise sales conversion."
    },
    {
        "rank": 17,
        "slug": "17_openai_workloads_inference_frontier",
        "company": "OpenAI",
        "role": "Strategic Finance, Research (Workloads)",
        "project_title": "Inference P99 Latency vs GPU Utilization Cost Efficiency Optimizer",
        "task_type": "regression",
        "model_family": "RF",
        "seed": 1017,
        "n_rows": 1200,
        "target": "effective_margin_per_m_tok",
        "unit": "$/M",
        "features": ["gpu_memory_util", "p99_latency_ms", "throughput_tok_s", "quantization_bits", "batch_size", "reserved_cap_share"],
        "domain_context": "Tier-1 sponsorship-supported Research Strategic Finance evaluating latency, throughput, and inference $/token economics ($234K-$325K).",
        "talking_point": "Gives Scaling & Inference engineers a single dollar-margin objective uniting P99 millisecond impact with GPU CapEx amortization."
    },
    {
        "rank": 18,
        "slug": "18_openai_generalist_deal_capex",
        "company": "OpenAI",
        "role": "Strategic Finance Generalist",
        "project_title": "Enterprise AI Commercial Deal & Multi-Year Compute CapEx ROI Simulator",
        "task_type": "regression",
        "model_family": "GBR",
        "seed": 1018,
        "n_rows": 1000,
        "target": "deal_3yr_npv_m",
        "unit": "$M",
        "features": ["contract_acv_m", "api_commit_pct", "capex_alloc_m", "compute_cost_deflation", "churn_prob", "discount_pct"],
        "domain_context": "Commercial transaction underwriting, capital markets & P&L/BS/CF ROI modeling ($234K-$260K SF, sponsorship supported).",
        "talking_point": "Underwrites 3-year strategic compute commitments factoring hardware depreciation deflation vs. contractual volume floors."
    },
    {
        "rank": 19,
        "slug": "19_hera_care_marketplace_integrity",
        "company": "Hera",
        "role": "Strategic Finance (Founding Team)",
        "project_title": "Two-Sided Care Marketplace Unit Margin & Claims Denial Risk Scorer",
        "task_type": "classification",
        "model_family": "RF_CLF",
        "seed": 1019,
        "n_rows": 1000,
        "target": "payer_claim_denial_flag",
        "unit": "prob",
        "features": ["caregiver_fill_rate", "medicaid_payer_tier", "evv_clock_mismatch", "prior_auth_missing", "msa_ramp_weeks", "billed_amt_usd"],
        "domain_context": "Tier-1 zero-gate NYC founding seat ($150K-$210K) owning care supply/demand unit economics and Hex/Claude revenue integrity.",
        "talking_point": "Predicts electronic-visit-verification and prior-auth leaks before claims submission so billed care converts cleanly to cash."
    },
    {
        "rank": 20,
        "slug": "20_alphasense_o2c_ar_delinquency",
        "company": "AlphaSense",
        "role": "Accounts Receivable Manager",
        "project_title": "Order-to-Cash 60+ Day AR Delinquency & Dunning Priority Risk Engine",
        "task_type": "classification",
        "model_family": "RF_CLF",
        "seed": 1020,
        "n_rows": 1200,
        "target": "delinquent_60d_flag",
        "unit": "prob",
        "features": ["invoice_amt_k", "days_to_renewal", "prior_late_count", "open_billing_disputes", "avg_hist_pay_days", "auto_pay_enabled"],
        "domain_context": "IPO-readiness O2C collections leadership in NetSuite/Salesforce preventing DSO creep while protecting NRR ($98K-$135K NYC/Remote).",
        "talking_point": "Separates disputed high-ACV enterprise renewals from routine dunning, isolating high-risk cash exposure systematically."
    },
]


def _generate_dataset(spec: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(spec["seed"])
    n = spec["n_rows"]
    feats = spec["features"]
    data: dict[str, np.ndarray] = {}

    for idx, f in enumerate(feats):
        fl = f.lower()
        if any(w in fl for w in ["flag", "missing", "auto_pay", "enabled", "multi_year", "auto_renew"]):
            data[f] = rng.binomial(1, 0.25 if "flag" in fl or "missing" in fl else 0.40, n).astype(float)
        elif any(w in fl for w in ["pct", "rate", "ratio", "share", "util", "retain", "attach", "mix", "attainment", "attain"]):
            lo, hi = (0.60, 1.20) if "attain" in fl or "nrr" in fl else (0.15, 0.92)
            data[f] = np.round(rng.uniform(lo, hi, n), 3)
        elif any(w in fl for w in ["days", "count", "cnt", "aes", "lives", "orders", "units", "weeks", "hours", "batch", "bits", "breaches"]):
            hi_int = 90 if "aes" in fl or "days" in fl else (120 if "units" in fl or "orders" in fl else 15)
            lo_int = 4 if "aes" in fl or "units" in fl else 0
            data[f] = rng.integers(lo_int, hi_int, n).astype(float)
        else:
            scale = 100.0 if "amt" in fl or "cost" in fl or "price" in fl or "usd" in fl or "capex" in fl or "acv" in fl else 25.0
            data[f] = np.round(rng.uniform(0.2 * scale, 1.8 * scale, n), 2)

    df = pd.DataFrame(data)
    # Build meaningful nonlinear signal using first 6 features cleanly normalized
    norm = (df - df.mean()) / (df.std().replace(0, 1.0))
    f0, f1, f2, f3, f4, f5 = [norm[c].values for c in feats[:6]]

    signal = (
        1.15 * f0
        + 0.85 * f1
        + 0.48 * f0 * f1  # genuine interaction so tree ensemble meaningfully beats linear baseline
        - 0.62 * f2
        + 0.45 * np.tanh(f3)
        + 0.35 * (f4 ** 2)
        - 0.40 * f5
    )

    if spec["task_type"] == "regression":
        # Anchor scale meaningfully to unit
        base_level = 120.0 if spec["unit"] in ("$K", "$", "$/M", "pts") else 22.0
        spread = 28.0 if spec["unit"] != "%" else 12.0
        target_vals = base_level + spread * signal + rng.normal(0, spread * 0.18, n)
        if spec["unit"] in ("$K", "$M", "$"):
            target_vals = np.maximum(2.0, target_vals)
        df[spec["target"]] = np.round(target_vals, 3)
    else:
        logits = -0.55 + 1.25 * signal + 0.30 * f0 * f2
        probs = 1.0 / (1.0 + np.exp(-logits))
        df[spec["target"]] = rng.binomial(1, probs)

    return df


def run_single_project(spec: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    df = _generate_dataset(spec)
    X = df[spec["features"]]
    y = df[spec["target"]]

    if spec["task_type"] == "regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=spec["seed"])
        scaler = StandardScaler()
        base = Ridge(alpha=1.0).fit(scaler.fit_transform(X_train), y_train)
        pred_b = base.predict(scaler.transform(X_test))

        if spec["model_family"] == "GBR":
            adv = GradientBoostingRegressor(n_estimators=160, max_depth=3, learning_rate=0.06, random_state=spec["seed"]).fit(X_train, y_train)
        else:
            adv = RandomForestRegressor(n_estimators=160, max_depth=10, random_state=spec["seed"]).fit(X_train, y_train)
        pred_a = adv.predict(X_test)

        b_r2 = float(r2_score(y_test, pred_b))
        a_r2 = float(r2_score(y_test, pred_a))
        b_mae = float(mean_absolute_error(y_test, pred_b))
        a_mae = float(mean_absolute_error(y_test, pred_a))
        a_rmse = float(root_mean_squared_error(y_test, pred_a))
        mae_reduction_pct = round(((b_mae - a_mae) / max(1e-9, b_mae)) * 100.0, 2)

        metrics = {
            "primary_metric_name": "R2",
            "baseline_primary": round(b_r2, 4),
            "ml_primary": round(a_r2, 4),
            "baseline_error_label": f"MAE ({spec['unit']})",
            "baseline_error": round(b_mae, 4),
            "ml_error": round(a_mae, 4),
            "ml_rmse": round(a_rmse, 4),
            "error_reduction_pct": mae_reduction_pct,
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=spec["seed"]
        )
        scaler = StandardScaler()
        base = LogisticRegression(max_iter=500, random_state=spec["seed"]).fit(scaler.fit_transform(X_train), y_train)
        p_b = base.predict_proba(scaler.transform(X_test))[:, 1]

        adv = RandomForestClassifier(
            n_estimators=160, max_depth=8, min_samples_leaf=6, random_state=spec["seed"]
        ).fit(X_train, y_train)
        p_a = adv.predict_proba(X_test)[:, 1]

        b_auc = float(roc_auc_score(y_test, p_b))
        a_auc = float(roc_auc_score(y_test, p_a))
        b_brier = float(brier_score_loss(y_test, p_b))
        a_brier = float(brier_score_loss(y_test, p_a))
        a_pr = float(average_precision_score(y_test, p_a))
        brier_red_pct = round(((b_brier - a_brier) / max(1e-9, b_brier)) * 100.0, 2)

        metrics = {
            "primary_metric_name": "ROC-AUC",
            "baseline_primary": round(b_auc, 4),
            "ml_primary": round(a_auc, 4),
            "baseline_error_label": "Brier Score",
            "baseline_error": round(b_brier, 4),
            "ml_error": round(a_brier, 4),
            "ml_pr_auc": round(a_pr, 4),
            "error_reduction_pct": brier_red_pct,
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    throughput = round(spec["n_rows"] / max(1e-4, elapsed_ms / 1000.0), 1)

    return {
        "rank": spec["rank"],
        "slug": spec["slug"],
        "company": spec["company"],
        "role": spec["role"],
        "project_title": spec["project_title"],
        "task_type": spec["task_type"],
        "model_family": spec["model_family"],
        "n_rows": spec["n_rows"],
        "test_rows": int(len(X_test)),
        "features": spec["features"],
        "target": spec["target"],
        "elapsed_ms": elapsed_ms,
        "throughput_rows_per_sec": throughput,
        "data_tag": "[SYN]",
        **metrics,
    }


def scaffold_project_dir(spec: dict[str, Any], result: dict[str, Any]) -> None:
    p_dir = PROJECTS_DIR / spec["slug"]
    p_dir.mkdir(parents=True, exist_ok=True)

    (p_dir / "config.json").write_text(json.dumps({"spec": spec, "verified_run": result}, indent=2), encoding="utf-8")

    runner_code = f'''"""Standalone CLI runner for #{spec["rank"]:02d} {spec["company"]} — {spec["project_title"]}."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from all_20_ml_projects import PROJECT_SPECS, run_single_project

if __name__ == "__main__":
    spec = next(s for s in PROJECT_SPECS if s["slug"] == "{spec["slug"]}")
    print(json.dumps(run_single_project(spec), indent=2))
'''
    (p_dir / "run_project.py").write_text(runner_code, encoding="utf-8")

    readme = f"""# Project #{spec['rank']:02d}: {spec['company']} — {spec['project_title']} `[SYN]`

- **Target Startup & Role:** **{spec['company']}** — *{spec['role']}*
- **Domain Context:** {spec['domain_context']}
- **ML Task:** `{spec['task_type'].upper()}` (`{spec['model_family']}` vs. linear baseline on `n={spec['n_rows']}` `[SYN]` rows, `25%` held-out test split)
- **Features:** `{', '.join(spec['features'])}`
- **Target Variable:** `{spec['target']}` (`{spec['unit']}`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | {result['primary_metric_name']} | {result['baseline_error_label']} | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `{result['baseline_primary']}` | `{result['baseline_error']}` | — | — |
| `{spec['model_family']}` ML Model | **`{result['ml_primary']}`** | **`{result['ml_error']}`** | **`{result['error_reduction_pct']}%`** | `{result['elapsed_ms']} ms` |

## Run Standalone
```bash
python projects/{spec['slug']}/run_project.py
```

## Interview Talking Point
> {spec['talking_point']}
"""
    (p_dir / "README.md").write_text(readme, encoding="utf-8")


def run_all_20() -> list[dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in PROJECT_SPECS:
        res = run_single_project(spec)
        scaffold_project_dir(spec, res)
        results.append(res)
    (REPORTS_DIR / "all_20_projects_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = run_all_20()
    print(f"Completed all {len(out)} startup ML projects!")
    for r in out:
        print(
            f"#{r['rank']:02d} {r['company']:<26} | {r['model_family']:<6} | "
            f"{r['primary_metric_name']}: base={r['baseline_primary']:.4f} -> ml={r['ml_primary']:.4f} | "
            f"ErrRed={r['error_reduction_pct']:+.2f}% | {r['elapsed_ms']}ms"
        )
