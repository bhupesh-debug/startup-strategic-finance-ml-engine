# Project #11: Maven Clinic — Employer Client Lives, Product Mix & Monthly BvA Forecast Model `[SYN]`

- **Target Startup & Role:** **Maven Clinic** — *Strategic Finance Associate*
- **Domain Context:** NYC Hybrid quantitative client/product revenue forecasting and Accounting close partnering ($94K-$110K + eq).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `enrolled_lives_k, pmpm_fee_usd, utilization_tier, enterprise_product_count, clinical_margin_pct, impl_lag_days`
- **Target Variable:** `client_monthly_rev_k` (`$K`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($K) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8921` | `13.1801` | — | — |
| `GBR` ML Model | **`0.9666`** | **`7.1906`** | **`45.44%`** | `220.99 ms` |

## Run Standalone
```bash
python projects/11_maven_clinic_client_rev_bva/run_project.py
```

## Interview Talking Point
> Links employer launch dates and product module attach directly to monthly GAAP BvA reviews.
