# Project #20: AlphaSense — Order-to-Cash 60+ Day AR Delinquency & Dunning Priority Risk Engine `[SYN]`

- **Target Startup & Role:** **AlphaSense** — *Accounts Receivable Manager*
- **Domain Context:** IPO-readiness O2C collections leadership in NetSuite/Salesforce preventing DSO creep while protecting NRR ($98K-$135K NYC/Remote).
- **ML Task:** `CLASSIFICATION` (`RF_CLF` vs. linear baseline on `n=1200` `[SYN]` rows, `25%` held-out test split)
- **Features:** `invoice_amt_k, days_to_renewal, prior_late_count, open_billing_disputes, avg_hist_pay_days, auto_pay_enabled`
- **Target Variable:** `delinquent_60d_flag` (`prob`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | ROC-AUC | Brier Score | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.806` | `0.1776` | — | — |
| `RF_CLF` ML Model | **`0.8216`** | **`0.1723`** | **`2.97%`** | `366.78 ms` |

## Run Standalone
```bash
python projects/20_alphasense_o2c_ar_delinquency/run_project.py
```

## Interview Talking Point
> Separates disputed high-ACV enterprise renewals from routine dunning, isolating high-risk cash exposure systematically.
