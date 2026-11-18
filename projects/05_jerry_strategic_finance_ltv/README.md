# Project #05: Jerry.ai — InsurTech Vertical CAC Payback, Policy Retention & LTV Model `[SYN]`

- **Target Startup & Role:** **Jerry.ai** — *Strategic Finance Manager*
- **Domain Context:** AI car-ownership & insurance marketplace underwriting across new verticals and paid channels ($150K-$225K Remote).
- **ML Task:** `REGRESSION` (`RF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `policy_premium_usd, carrier_take_rate, month6_retention, cac_usd, claim_assist_score, multi_product_attach`
- **Target Variable:** `cohort_24m_ltv_usd` (`$`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8994` | `12.6343` | — | — |
| `RF` ML Model | **`0.9025`** | **`12.5597`** | **`0.59%`** | `435.06 ms` |

## Run Standalone
```bash
python projects/05_jerry_strategic_finance_ltv/run_project.py
```

## Interview Talking Point
> Maps Bankers Life insurance sensitivity analytics onto consumer policy persistence, carrier commission mix, and Month-12 payback gates.
