# Project #09: Zocdoc — Patient Marketplace Booking Volume, Take-Rate & IR Bridge Engine `[SYN]`

- **Target Startup & Role:** **Zocdoc** — *Senior Manager, Strategic Finance & IR*
- **Domain Context:** NYC Hybrid Strategic Finance & IR bridge decomposing volume beats vs. enterprise health-system take-rate mix.
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `patient_bookings_k, enterprise_system_mix, effective_take_rate_usd, provider_churn_pct, rebook_share, cac_efficiency`
- **Target Variable:** `net_marketplace_rev_k` (`$K`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($K) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8969` | `11.6327` | — | — |
| `GBR` ML Model | **`0.9544`** | **`8.1574`** | **`29.88%`** | `261.41 ms` |

## Run Standalone
```bash
python projects/09_zocdoc_ir_booking_take_rate/run_project.py
```

## Interview Talking Point
> Produces investor-grade volume/mix/rate bridges explaining why lower enterprise take-rate expands multi-year LTV.
