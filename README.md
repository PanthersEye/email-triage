# Email Triage & Reply API

A production-style FastAPI app that classifies inbound emails (support/sales/hr/finance/general), estimates priority, and optionally drafts replies with OpenAI (if configured).

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optionally set OPENAI_API_KEY for /reply


