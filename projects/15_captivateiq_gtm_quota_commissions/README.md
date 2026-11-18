# Project #15: CaptivateIQ — GTM Sales Capacity, AE Ramp & Commission Accelerator Forecaster `[SYN]`

- **Target Startup & Role:** **CaptivateIQ** — *Strategic Finance Manager, GTM*
- **Domain Context:** E-Verify Tier-1 Remote GTM Strategic Finance owning capacity planning, net new ARR, CaptivateIQ commissions & Adaptive.
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `ramping_aes, quota_attainment, pipeline_coverage, accelerator_mult, nrr, sales_cycle_days`
- **Target Variable:** `net_new_arr_m` (`$M`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($M) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8109` | `15.3893` | — | — |
| `GBR` ML Model | **`0.9519`** | **`7.5751`** | **`50.78%`** | `268.46 ms` |

## Run Standalone
```bash
python projects/15_captivateiq_gtm_quota_commissions/run_project.py
```

## Interview Talking Point
> Links AE ramp tenure curves and Salesforce pipeline coverage directly into net new ARR and marginal commission cost-of-ARR.
