#!/usr/bin/env bash
set -e
export PYTHONPATH=.
uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" ${RELOAD:+--reload}
