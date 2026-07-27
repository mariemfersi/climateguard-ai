# ClimateGuard AI

> An agentic, explainable reinsurance pricing & portfolio-steering platform for
> climate-adjusted catastrophe risk.

**Status:** 🚧 Phase 0 — Foundations & Environment Setup (see `docs/environment_setup.md`)

This README is intentionally minimal right now. It will be filled in progressively
as each phase of the [Implementation Roadmap](docs/implementation_roadmap.md)
completes, finishing with the full structure defined in the roadmap's Phase 11
(pitch, architecture diagram, quickstart, demo link).

## Quickstart (current: local dev only)

```bash
git clone <this-repo>
cd climateguard-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in API keys as you reach the phases that need them
pytest
```

## Project structure

See each folder's own `README.md` for its purpose, or the full architecture
document at `docs/`.

## Documentation

- `docs/environment_setup.md` — how to get a working local environment
- `docs/data_sources.md` — external data sources and licensing (added in Phase 1)
- `docs/synthetic_data_methodology.md` — synthetic exposure/claims methodology (added in Phase 2)
- More docs are added as each phase completes — see the roadmap.
