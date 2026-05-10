# AGENTS.md

## Cursor Cloud specific instructions

### Overview

GuardianRail is a Python-only Streamlit app (no frontend build step, no Docker, no external databases). The app has two backends: **mock** (default, no GPU) and **real** (AMD MI300X GPU required). Cloud agents should always use mock mode.

### Running the app

```bash
source /workspace/.venv/bin/activate
streamlit run frontend/app.py --server.headless=true --server.port=8501
```

The app runs at `http://localhost:8501`. Mock mode is the default (`GUARDIAN_BACKEND=mock` or unset). Do NOT set `GUARDIAN_BACKEND=real` — there is no GPU available.

### Linting

No linting config is committed to the repo. Use `ruff check src/ frontend/` for basic linting. The only existing warnings are E402 (import ordering) in `src/mock_guardian.py` — these are pre-existing.

### Testing

There is no test suite in the repo. To verify the app works, start Streamlit and click the three demo prompt buttons (Normal, Prompt Injection, Social Engineering). Confirm audit log entries appear in the SQLite database at `artifacts/guardianrail.sqlite3`.

### Key caveats

- The virtual environment must be at `/workspace/.venv`. The update script creates it if missing.
- `requirements-local.txt` is the correct dependency file for mock mode (minimal: torch, tqdm, streamlit). `requirements.txt` is for the GPU backend and pulls in large HuggingFace libraries unnecessarily.
- SQLite database files under `artifacts/` are gitignored. They are auto-created on first run.
- All source code is in `src/` and `frontend/`. Scripts in `scripts/` are standalone analysis utilities, not part of the main app.
