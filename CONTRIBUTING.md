# Contributing Guidelines

1. **Reproducibility & Leakage Prevention**: All dataset synthesis (`[SYN]`) and model splits must keep fixed random seeds and fit preprocessors strictly inside `X_train` before transforming `X_test`.
2. **Evidence Tagging**: Label synthetic benchmarks `[SYN]`, simulated outputs `[SIMULATED]`, and verified local runs `[VERIFIED]`. Never fabricate empirical results or credentials.
3. **Testing Gate**: Run `pytest tests/ -v` across all 8 test suites before committing or opening a pull request.
4. **Security Gate**: Zero hardcoded tokens, passwords, or unsafe deserialization (`eval`, `exec`, `pickle.load`).
