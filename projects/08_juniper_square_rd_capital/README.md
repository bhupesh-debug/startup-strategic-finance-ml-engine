# Project #08: Juniper Square — R&D Engineering Stage-Gate ROI & ASC 350-40 Capitalization Model `[SYN]`

- **Target Startup & Role:** **Juniper Square** — *Finance Director, R&D*
- **Domain Context:** CTO/CPO Finance partnering across US & India R&D hubs, Pigment/TeamOhana headcount, cloud AI cost & cap-vs-expense ($175K-$220K).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `eng_pod_cost_k, india_us_blend_ratio, cloud_ai_compute_k, asc350_cap_pct, arr_unlock_k, sprint_velocity_idx`
- **Target Variable:** `product_net_roi_pct` (`%`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE (%) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8912` | `5.9603` | — | — |
| `GBR` ML Model | **`0.9662`** | **`3.3135`** | **`44.41%`** | `206.44 ms` |

## Run Standalone
```bash
python projects/08_juniper_square_rd_capital/run_project.py
```

## Interview Talking Point
> Aligns Engineering squad velocity and blended US/India cost efficiency with auditable ASC 350-40 software capitalization gates.
