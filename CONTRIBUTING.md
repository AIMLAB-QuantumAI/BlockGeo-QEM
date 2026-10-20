# Contributing to BlockGeo-QEM

Bug reports, questions and pull requests are welcome. This page describes how to report a problem and how to propose a change.

## Reporting a problem

Open an issue with the **Bug report** template. Please include

- the command you ran and the full error message,
- the `BLOCKGEO_*` environment variables you set, if any,
- your operating system, Python version and the versions of the main packages, for example the output of
  `python -c "import numpy, scipy, torch, qiskit, qiskit_aer; print(numpy.__version__, scipy.__version__, torch.__version__, qiskit.__version__, qiskit_aer.__version__)"`.

For a question or an idea, use the **Feature request** template or a blank issue.

## Proposing a change

1. Fork the repository and create a branch from `main`.
2. Make the change. Keep scripts runnable from the repository root with `python src/<script>.py`, and take every path from `src/paths.py` instead of writing machine-specific paths.
3. Run the tests with `pytest -q`. They take about a minute on a CPU and use temporary directories only.
4. Add a test when you fix a bug or add a feature.
5. Open a pull request against `main`. Describe what changes and why, and link the issue it addresses with `Closes #<number>`.

The CI workflow runs the tests on every pull request.

## Changes that may affect results

Many changes should leave every number unchanged, for example refactoring, documentation, logging or error handling. For these, compare the outputs before and after the change on a small problem, such as the quick start in the README, and say so in the pull request. If a change alters results on purpose, explain the reason and add an entry to `CHANGELOG.md`.

## Style

Follow the style of the surrounding code. Comments and docstrings explain what the code computes and why, in plain English.
