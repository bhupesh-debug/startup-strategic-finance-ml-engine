# Project #12: FORT Robotics — ASC 606 Multi-Element SSP & SOX 302/404 Control Exception Scorer `[SYN]`

- **Target Startup & Role:** **FORT Robotics** — *Head of SEC Reporting*
- **Domain Context:** Public-company external reporting readiness, SOX 302/404 audit controls & Workiva XBRL close validation ($140K-$220K).
- **ML Task:** `CLASSIFICATION` (`RF_CLF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `ssp_deviation_pct, manual_je_late_flag, sod_conflict_score, multi_elem_bundle_cnt, subledger_tieout_diff_k, cut_off_days`
- **Target Variable:** `sox_control_exception_flag` (`prob`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | ROC-AUC | Brier Score | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8223` | `0.173` | — | — |
| `RF_CLF` ML Model | **`0.8196`** | **`0.1739`** | **`-0.49%`** | `388.55 ms` |

## Run Standalone
```bash
python projects/12_fort_robotics_sec_sox_controls/run_project.py
```

## Interview Talking Point
> Demonstrates automated pre-audit detection of Segregation-of-Duties (SoD) and SSP allocation risk (framed transparently under CPA-candidate track).
