# Project #07: Jerry.ai — Marketplace OpEx Variance Flux & Vendor Spend Anomaly Classifier `[SYN]`

- **Target Startup & Role:** **Jerry.ai** — *Finance Manager*
- **Domain Context:** Operational FP&A close control, Ramp/NetSuite spend underwriting, and monthly vendor flux escalation ($150K-$225K Remote).
- **ML Task:** `CLASSIFICATION` (`RF_CLF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `vendor_spend_ratio, po_missing_flag, dup_seat_pct, dept_headcount_delta, contract_auto_renew, prior_flux_breaches`
- **Target Variable:** `unbudgeted_overrun_flag` (`prob`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | ROC-AUC | Brier Score | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8544` | `0.1568` | — | — |
| `RF_CLF` ML Model | **`0.8145`** | **`0.1755`** | **`-11.99%`** | `301.79 ms` |

## Run Standalone
```bash
python projects/07_jerry_finance_mgr_opex_flux/run_project.py
```

## Interview Talking Point
> Flags duplicate SaaS seats, missing POs, and auto-renewal traps before month-end close locks OpEx overspend.
