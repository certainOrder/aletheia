# Copilot/LMM Contributor Instructions

This document sets expectations for AI assistants (and humans) contributing to this repository. The goals are: professionalism, security, maintainability, and a consistent developer experience.

## Operating Principles
- Be explicit: state assumptions briefly when details are missing; prefer safe defaults.
- Professional tone: concise, actionable, reproducible steps.
- Security-first: never commit secrets or call external services unless explicitly required.
- Least-change: prefer minimal diffs that satisfy the requirement and preserve style.
- Reproducible: ensure changes build, lint, typecheck, and test before submitting.

## Code Quality Gates
- Lint: `ruff` (strict) — line length 100, rules: `E,F,I,UP`.
- Format: `black` and `ruff format`.
- Types: `mypy` — strictish; ignore missing imports is allowed.
- Tests: `pytest` with coverage gate (≥85%).
- Hooks: pre-commit must pass locally.

Use the Makefile:
```bash
make setup      # venv + dev deps + hooks
make format     # ruff --fix + ruff format + black
make lint       # ruff check
make typecheck  # mypy
make test       # pytest with coverage gate
```

## Project Conventions
- Language: Python 3.11.
- Framework: FastAPI + SQLAlchemy 2.0 typed ORM + pgvector.
- Config via `.env` (do not commit real secrets). See `.env.example`.
- DEV fallbacks: keep deterministic fallbacks for embeddings/chat to enable offline tests.
- SQLAlchemy models: use `Mapped[...]` annotations and `DeclarativeBase`.
- Public APIs: avoid breaking changes; document behavior in `README.md` and docs.

## Security & Secrets
- Never commit API keys, tokens, or credentials.
- Respect `.gitignore`; do not add secret files.
- For examples, use placeholders and document required env variables.
- No network calls in tests unless explicitly requested; prefer mocks/fallbacks.

## Testing Guidance
- Tests should be deterministic, isolated, and run offline.
- Use dependency overrides and dummy sessions for DB access.
- Cover happy paths and 1–2 edge cases; keep tests fast.
- Ensure total coverage ≥85% and that tests pass locally before committing.

## Branching & Commits
- Use feature branches; keep commits scoped and messages descriptive.
- Reference the feature or fix in the commit message.
- After edits: run `make format lint typecheck test`.

## Docker & Dev Env
- For full stack use:
```bash
cp .env.example .env
make compose-up      # start db, api, openwebui
make compose-logs    # tail logs
make compose-down    # stop and clean volumes
```

## File/Folder Hygiene
- Do not create top-level clutter; place docs under `docs/`.
- Tests under `tests/`, not mixed with app code.
- Prefer small, well-named modules and functions.

## Acceptable Dependencies
- Keep dependencies minimal and pinned via `requirements.txt` and `requirements-dev.txt`.
- Before adding a dep, justify the need and consider stdlib alternatives.

## When Details Are Missing
- Make 1–2 reasonable assumptions; note them in the PR description.
- Prefer reversible changes; avoid large refactors without prior plan/docs.

## PR Checklist (must pass)
- [ ] Code compiles; no syntax/type errors
- [ ] `make format lint typecheck test` all pass locally
- [ ] No secrets or hardcoded credentials
- [ ] Updated docs/tests as needed
- [ ] Minimal, focused diff; aligned with repo conventions

## Out of Scope / Do Not Do
- No speculative features without a plan
- No disabling/relaxing lints or coverage gates to pass checks
- No pushing broken builds

## Contacts
- Repository: this doc + `README.md` + `docs/` are the source of truth
- For ambiguity, prefer comments/docs and minimal, reversible changes
