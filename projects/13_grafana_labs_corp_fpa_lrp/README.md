# Project #13: Grafana Labs — Consolidated 3-Statement LRP & Rule-of-40 Public-Readiness Engine `[SYN]`

- **Target Startup & Role:** **Grafana Labs** — *Senior Director, Corporate FP&A*
- **Domain Context:** Enterprise SaaS Corporate FP&A consolidated P&L/BS/CF forecasting in Pigment and close-to-insight rhythm ($268K-$322K OTE).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `arr_yoy_growth_pct, fcf_margin_pct, ndr_pct, sm_pct_rev, rd_pct_rev, tax_eff_rate`
- **Target Variable:** `rule_of_40_score` (`pts`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE (pts) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8856` | `14.5166` | — | — |
| `GBR` ML Model | **`0.9698`** | **`7.3763`** | **`49.19%`** | `277.57 ms` |

## Run Standalone
```bash
python projects/13_grafana_labs_corp_fpa_lrp/run_project.py
```

## Interview Talking Point
> Quantifies trade-offs between top-line ARR growth investment and FCF margin levers across long-range plans.
