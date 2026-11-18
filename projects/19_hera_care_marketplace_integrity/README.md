# Project #19: Hera — Two-Sided Care Marketplace Unit Margin & Claims Denial Risk Scorer `[SYN]`

- **Target Startup & Role:** **Hera** — *Strategic Finance (Founding Team)*
- **Domain Context:** Tier-1 zero-gate NYC founding seat ($150K-$210K) owning care supply/demand unit economics and Hex/Claude revenue integrity.
- **ML Task:** `CLASSIFICATION` (`RF_CLF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `caregiver_fill_rate, medicaid_payer_tier, evv_clock_mismatch, prior_auth_missing, msa_ramp_weeks, billed_amt_usd`
- **Target Variable:** `payer_claim_denial_flag` (`prob`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | ROC-AUC | Brier Score | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8583` | `0.1534` | — | — |
| `RF_CLF` ML Model | **`0.8532`** | **`0.1573`** | **`-2.53%`** | `343.57 ms` |

## Run Standalone
```bash
python projects/19_hera_care_marketplace_integrity/run_project.py
```

## Interview Talking Point
> Predicts electronic-visit-verification and prior-auth leaks before claims submission so billed care converts cleanly to cash.
