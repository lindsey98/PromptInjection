# Contributing

Thanks for your interest in improving DRIP! This guide covers the essentials.

## Development setup

1. Create the environment (see the [README](./README.md#setup)):

   ```bash
   bash setup_env.sh
   conda activate prompt
   ```

2. Install the development tooling:

   ```bash
   pip install pytest ruff
   ```

## Before opening a pull request

- **Lint** with the shared configuration:

  ```bash
  ruff check .
  ```

- **Run the unit tests:**

  ```bash
  pytest
  ```

  GPU/model tests are skipped automatically when `torch` is unavailable, so the
  suite runs quickly on CPU-only machines.

- Keep changes focused, and write a clear PR description (what changed and why).
- Match the style of the surrounding code. New comments and docstrings should be
  in English.

## Reporting issues

Please open an issue with a minimal reproduction: the exact command you ran, and
the observed vs. expected behavior.

## Code of Conduct

All participants are expected to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).
