#!/usr/bin/env bash
set -euo pipefail
export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://emailtriage:emailtriage@localhost:5432/emailtriage}"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
