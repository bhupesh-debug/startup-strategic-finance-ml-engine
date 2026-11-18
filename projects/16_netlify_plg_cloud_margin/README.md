# Project #16: Netlify [Archived Template] — PLG Free-to-Enterprise Conversion & Hosting Bandwidth Margin Classifier `[SYN]`

- **Target Startup & Role:** **Netlify [Archived Template]** — *Senior Strategic Finance Manager*
- **Domain Context:** Archived Greenhouse template preserved for Developer Cloud PLG-to-Enterprise PQL conversion & compute margin caps.
- **ML Task:** `CLASSIFICATION` (`RF_CLF` vs. linear baseline on `n=900` `[SYN]` rows, `25%` held-out test split)
- **Features:** `team_seats_active, monthly_bandwidth_gb, ci_build_mins, sso_security_hits, days_since_signup, billing_card_flag`
- **Target Variable:** `enterprise_upgrade_flag` (`prob`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | ROC-AUC | Brier Score | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.847` | `0.1569` | — | — |
| `RF_CLF` ML Model | **`0.8486`** | **`0.1595`** | **`-1.65%`** | `412.44 ms` |

## Run Standalone
```bash
python projects/16_netlify_plg_cloud_margin/run_project.py
```

## Interview Talking Point
> Identifies PQL signals where free compute consumption justifies proactive Enterprise sales conversion.
