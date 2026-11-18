# Project #17: OpenAI — Inference P99 Latency vs GPU Utilization Cost Efficiency Optimizer `[SYN]`

- **Target Startup & Role:** **OpenAI** — *Strategic Finance, Research (Workloads)*
- **Domain Context:** Tier-1 sponsorship-supported Research Strategic Finance evaluating latency, throughput, and inference $/token economics ($234K-$325K).
- **ML Task:** `REGRESSION` (`RF` vs. linear baseline on `n=1200` `[SYN]` rows, `25%` held-out test split)
- **Features:** `gpu_memory_util, p99_latency_ms, throughput_tok_s, quantization_bits, batch_size, reserved_cap_share`
- **Target Variable:** `effective_margin_per_m_tok` (`$/M`)

## Observed Out-of-Sample Results (`VERIFIED` `[SYN]`)

| Model | R2 | MAE ($/M) | Error Reduction | Exec Latency |
|---|---:|---:|---:|---:|
| Linear Baseline | `0.8861` | `12.847` | — | — |
| `RF` ML Model | **`0.9103`** | **`11.4767`** | **`10.67%`** | `566.26 ms` |

## Run Standalone
```bash
python projects/17_openai_workloads_inference_frontier/run_project.py
```

## Interview Talking Point
> Gives Scaling & Inference engineers a single dollar-margin objective uniting P99 millisecond impact with GPU CapEx amortization.
