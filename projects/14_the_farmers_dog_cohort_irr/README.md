# Project #14: The Farmer's Dog — Subscription vs Retail Multi-Channel Cohort & IRR Guardrail Model `[SYN]`

- **Target Startup & Role:** **The Farmer's Dog** — *Director, Finance*
- **Domain Context:** NYC SoHo Director of Finance modeling multi-product/multi-channel promo guardrails, cohort payback, and IRR ($178K-$200K).
- **ML Task:** `REGRESSION` (`RF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `promo_discount_pct, spoilage_shipping_pct, sub_month6_retain, retail_halo_lift, cac_payback_mo, aov_usd`
- **Target Variable:** `channel_18m_irr_pct` (`%`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE (%) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8682` | `5.9499` | — | — |
| `RF` ML Model | **`0.8998`** | **`5.0301`** | **`15.46%`** | `515.26 ms` |

## Run Standalone
```bash
python projects/14_the_farmers_dog_cohort_irr/run_project.py
```

## Interview Talking Point
> Separates true incremental multi-channel IRR from promo-discounted low-LTV acquisition cohorts.
