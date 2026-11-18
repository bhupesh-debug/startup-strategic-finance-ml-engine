# Project #10: Vendelux — HubSpot-Maxio-QBO Three-Way Reconciliation & Series C Audit Scorer `[SYN]`

- **Target Startup & Role:** **Vendelux** — *FP&A Manager*
- **Domain Context:** Series B NYC SaaS FP&A unifying HubSpot bookings, Maxio subscriptions & QuickBooks Online GAAP ledger ($140K-$200K + eq).
- **ML Task:** `CLASSIFICATION` (`RF_CLF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `crm_maxio_amt_diff_pct, start_date_lag_days, custom_ramp_flag, qbo_deferred_mismatch, manual_adj_count, multi_year_contract`
- **Target Variable:** `recon_break_flag` (`prob`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | ROC-AUC | Brier Score | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.9014` | `0.1286` | — | — |
| `RF_CLF` ML Model | **`0.8705`** | **`0.1478`** | **`-14.94%`** | `407.51 ms` |

## Run Standalone
```bash
python projects/10_vendelux_crm_maxio_qbo_tieout/run_project.py
```

## Interview Talking Point
> Eliminates fundraising data-room breaks by catching CRM-to-Maxio start-date lags and custom ramp schedules before close.
