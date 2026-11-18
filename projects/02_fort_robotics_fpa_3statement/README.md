# Project #02: FORT Robotics — Hardware/Software/Services Blended Margin & 13-Wk Cash Model `[SYN]`

- **Target Startup & Role:** **FORT Robotics** — *FP&A Manager*
- **Domain Context:** Public-readiness / de-SPAC 3-statement driver model bridging robotics hardware inventory turns and ratable software ARR.
- **ML Task:** `REGRESSION` (`RF` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `hardware_units, sw_attach_pct, bom_cost_index, inventory_days, deferred_rev_ratio, headcount_ramp`
- **Target Variable:** `operating_cash_flow_k` (`$K`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($K) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.896` | `12.7701` | — | — |
| `RF` ML Model | **`0.9239`** | **`11.0656`** | **`13.35%`** | `369.79 ms` |

## Run Standalone
```bash
python projects/02_fort_robotics_fpa_3statement/run_project.py
```

## Interview Talking Point
> Prevents working-capital cash traps by tying BOM inventory lead times directly into the 13-week cash bridge alongside ASC 606 mix.
