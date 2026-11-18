# Project #03: Covera Health — Care Coordination Scan Volume & FTE Staffing Capacity Optimizer `[SYN]`

- **Target Startup & Role:** **Covera Health** — *Manager, FP&A*
- **Domain Context:** Workforce staffing and cost-per-order unit economics for clinical Care Coordination ops (NYC optional).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `weekly_scan_orders, fte_utilization, agent_ramp_share, attrition_pct, ai_triage_adoption, offshore_mix`
- **Target Variable:** `cost_per_scan_usd` (`$`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8769` | `14.0526` | — | — |
| `GBR` ML Model | **`0.9605`** | **`7.8858`** | **`43.88%`** | `235.65 ms` |

## Run Standalone
```bash
python projects/03_covera_health_care_capacity/run_project.py
```

## Interview Talking Point
> Quantifies cost-per-scan elasticity under seasonal demand spikes, agent attrition cohorts, and AI triage routing.
