# Refactor Guardrails

- Existing 15-fold results are historical development evidence and read-only.
- Never overwrite existing experiment artifacts.
- Never change data files during refactor.
- Never silently change labels, folds, seeds, hyperparameters, or dependencies.
- No scientific behavior change is allowed unless explicitly requested in a later prompt.
- Primary V1 run provenance must not claim historical-run reproduction or clinical threshold validation.
- Each refactor step must have a narrow allowed-file list.
- Each step must pass its requested tests before proceeding.
- No full Colab/scientific training run during structural refactor.
- Product/UI clinical-report code must remain separate from the research core.
- Legacy code must be preserved until its replacement has passed explicit tests.
