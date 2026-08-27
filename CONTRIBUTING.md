# Contributing

This repository is a **personal portfolio / teaching demo**, not an open product with a contributor ladder. Small, well-scoped changes are still welcome.

## Before you open a PR

1. Fork and branch from `main`.
2. Keep the Streamlit entry file named `historajstreamlit.py` unless you also update Codespaces / Cloud instructions.
3. Put parsing, filters, stats, layout, and matplotlib in `historaj/` — keep `historajstreamlit.py` as UI.
4. Add pytest coverage for logic that does not need a browser or the network.
5. Do not commit `.venv/`, `.env`, or `.streamlit/secrets.toml`.

## Local checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
streamlit run historajstreamlit.py
```

## What belongs here

- Bug fixes in loaders, filters, stats, or plot scaling
- Tests and documentation that stay honest about limits
- Dependency pins required to keep CI green

## What does not

- Rewriting this into a general-purpose statistics platform
- Issue/PR templates aimed at a large external community
- Secrets, sample datasets with confidential records, or a fake “live demo” URL
