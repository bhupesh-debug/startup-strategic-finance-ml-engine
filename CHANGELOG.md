# Changelog

All notable changes to `startup-strategic-finance-ml-engine` are documented in this file.

## [1.1.0] - 2026-11-17
### Added
- Scaffolded all **20 startup-specific ML projects** under `projects/01_glide_ai_billing_forecast` through `projects/20_alphasense_o2c_ar_delinquency` with standalone `run_project.py`, `config.json`, and `README.md`.
- Implemented `src/all_20_ml_projects.py` evaluating linear baselines (`Ridge` / `LogisticRegression`) vs. tree ensembles (`GBR` / `RF`) with leakage-free train/test splits and latency benchmarking.
- Added `src/quant_research_and_montecarlo.py` covering 5 formal statistical hypothesis tests (`H1–H5`), `10,000-trial` Monte Carlo simulation (`VaR 95%`, `CVaR 95%`), Base/Best/Worst scenarios summing to `100%`, Tornado sensitivity (`1% Pareto factor`), static security audit, and charts `fig5–fig8`.
- Added 8-level verification suite in `tests/test_all_20_projects.py` and GitHub Actions CI workflow in `.github/workflows/ci.yml`.
- Generated complete 20-Section `/repo` Release Report (`reports/REPO_20_SECTION_RELEASE_REPORT.md` & PDF).

## [1.0.0] - 2026-11-17
### Added
- Initial 3-module core pipeline (`src/startup_finance_ml_pipeline.py`) covering SaaS ARR/Burn, AI GPU Compute Margin, and O2C AR Delinquency Risk (`pytest` 4/4 passed).
