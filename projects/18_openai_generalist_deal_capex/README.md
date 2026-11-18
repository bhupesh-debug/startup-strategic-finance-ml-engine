# Project #18: OpenAI — Enterprise AI Commercial Deal & Multi-Year Compute CapEx ROI Simulator `[SYN]`

- **Target Startup & Role:** **OpenAI** — *Strategic Finance Generalist*
- **Domain Context:** Commercial transaction underwriting, capital markets & P&L/BS/CF ROI modeling ($234K-$260K SF, sponsorship supported).
- **ML Task:** `REGRESSION` (`GBR` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `contract_acv_m, api_commit_pct, capex_alloc_m, compute_cost_deflation, churn_prob, discount_pct`
- **Target Variable:** `deal_3yr_npv_m` (`$M`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($M) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8135` | `14.5035` | — | — |
| `GBR` ML Model | **`0.9474`** | **`7.1893`** | **`50.43%`** | `296.43 ms` |

## Run Standalone
```bash
python projects/18_openai_generalist_deal_capex/run_project.py
```

## Interview Talking Point
> Underwrites 3-year strategic compute commitments factoring hardware depreciation deflation vs. contractual volume floors.
