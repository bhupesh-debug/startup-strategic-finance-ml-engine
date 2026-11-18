# Project #01: Glide — AI Contract-to-Cash Billing & Cash Runway Forecaster `[SYN]`

- **Target Startup & Role:** **Glide** — *Strategic Finance Lead*
- **Domain Context:** Community-bank embedded fintech contract billing, AI auto-reconciliation, and monthly runway cash collection ($140K-$180K + eq NYC).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `fintech_contracts, usage_overage_pct, auto_recon_rate, days_to_invoice, cogs_alloc_ratio, churn_rate`
- **Target Variable:** `net_cash_collected_k` (`$K`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($K) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8769` | `13.1642` | — | — |
| `GBR` ML Model | **`0.9607`** | **`7.4769`** | **`43.2%`** | `368.29 ms` |

## Run Standalone
```bash
python projects/01_glide_ai_billing_forecast/run_project.py
```

## Interview Talking Point
> Models nonlinear impact of contract-to-cash invoice lag and usage overages on monthly cash runway with automated SQL billing tie-out.
