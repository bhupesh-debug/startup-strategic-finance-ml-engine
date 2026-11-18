# Project #06: Baseten — AI Inference GPU Cluster Utilization & Gross Margin Optimizer `[SYN]`

- **Target Startup & Role:** **Baseten** — *Strategic Finance*
- **Domain Context:** Usage-based AI inference revenue, GPU utilization efficiency, enterprise deal minimums & margin analysis ($200K-$280K NYC).
- **ML Task:** `REGRESSION` (`RF` vs. linear baseline on `n=1000` `[SYN]` rows, `25%` held-out test split)
- **Features:** `gpu_utilization, reserved_share, tokens_k_per_min, realized_price_m_tok, batch_size, p99_latency_ms`
- **Target Variable:** `inference_gross_margin_pct` (`%`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE (%) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8736` | `5.8717` | — | — |
| `RF` ML Model | **`0.9291`** | **`4.2718`** | **`27.25%`** | `579.77 ms` |

## Run Standalone
```bash
python projects/06_baseten_gpu_compute_margin/run_project.py
```

## Interview Talking Point
> Pinpoints exact GPU utilization thresholds and spot/reserved cluster splits needed to protect >=65% inference gross margins.
