# Project #04: Narvar — B2B SaaS + Transactional Shipping Volume & NRR Revenue Engine `[SYN]`

- **Target Startup & Role:** **Narvar** — *Senior Manager, FP&A*
- **Domain Context:** Post-purchase SaaS + peak Q4 transactional parcel volume forecasting and AOP/LRP modeling ($157K-$225K Remote).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `base_saas_arr_m, holiday_ship_mult, nrr_rate, take_rate_bps, ramping_aes, quota_attain`
- **Target Variable:** `quarterly_net_new_rev_k` (`$K`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($K) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8962` | `13.5427` | — | — |
| `GBR` ML Model | **`0.9599`** | **`7.9057`** | **`41.62%`** | `249.99 ms` |

## Run Standalone
```bash
python projects/04_narvar_saas_transactional_arr/run_project.py
```

## Interview Talking Point
> Decouples deterministic base SaaS NRR from volatile merchant shipping GMV elasticity so BvA variances stay transparent.
