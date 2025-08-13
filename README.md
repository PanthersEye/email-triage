# Email Triage & Reply API

Here’s a one-page, plain-text write-up you can paste into the README:

---

# Overview

This project is a small, production-style **FastAPI** application that triages incoming emails. It exposes documented HTTP endpoints to (1) **train** a text classifier, (2) **predict** a category for an email, (3) **score priority** using simple keyword heuristics, and (4) optionally **draft a reply** via OpenAI (if an API key is configured). The service is framework-agnostic on the client side—anything that can make HTTP requests can use it.

# What It Does

1. **Email Classification**

   * Combines `subject` + `body`, vectorizes with **TF-IDF**, and classifies using **Logistic Regression** into one of five labels: `support`, `sales`, `hr`, `finance`, `general`.
   * Model artifacts (vectorizer + classifier) are persisted under `model/` as `.joblib` files so you don’t have to retrain every run.

2. **Priority Scoring**

   * Applies a lightweight keyword heuristic (e.g., “urgent”, “asap”, “emergency”, “issue”, “can’t access”) to return a **priority score** in `[0,1]` and a **priority label** (`Low` / `Medium` / `High`). This can be used to route or escalate emails quickly.

3. **Reply Drafting (Optional)**

   * If `OPENAI_API_KEY` is present, `/reply` uses OpenAI Chat Completions to generate a concise, professional reply body given the original email and optional guidance.
   * If no key is set, the endpoint returns a harmless placeholder message so the rest of the API remains fully functional.

# Key Endpoints (HTTP)

* `GET /health` – Basic health check.
* `POST /train_synth` – Train on generated examples (JSON body `{ "n_per_cat": 12 }`, default 12 per class). Good for demo/prototyping.
* `POST /train` – Train on a **CSV upload** (multipart) with columns: `subject, body, category`. Use this to train on real data.
* `POST /predict` – Predict a single email’s category; returns class probabilities and priority score/label.
* `POST /batch_predict` – Predict for a list of emails in one call.
* `POST /reply` – Draft a professional reply (uses OpenAI if configured).

Swagger docs at `/docs` include typed schemas and examples.

# Code Layout

* `app/main.py` – App factory + metadata (title, description, tags, contact, license) and router mounting.
* `app/routers/classifier.py` – All HTTP endpoints. Minimal controller logic; delegates to services.
* `app/schemas.py` – Pydantic models for requests/responses; drives validation and OpenAPI docs.
* `app/services/model.py` – ML pipeline (TF-IDF vectorizer, Logistic Regression), training, prediction, artifact persistence, thread-safe access.
* `app/services/priority.py` – Keyword-based priority scoring and labeling.
* `app/core/config.py` – Environment-based settings via `pydantic-settings` (`OPENAI_API_KEY`, `OPENAI_MODEL`, `HOST`, `PORT`, etc.).
* `model/` – Saved artifacts (`vectorizer.joblib`, `model.joblib`, test splits) that allow warm starts.

# Data Flow

1. **Train**

   * `/train_synth` generates synthetic examples per label; or `/train` ingests a labeled CSV.
   * Text is combined (`subject + body`) → TF-IDF → Logistic Regression fit.
   * Artifacts saved to `model/`. Future server runs auto-load these if present.

2. **Predict**

   * Client sends `subject` and `body`.
   * Service loads artifacts, transforms text, returns:

     * `category` (argmax),
     * `probabilities` per class,
     * `priority_score` and `priority_label`.

3. **Reply (optional)**

   * If OpenAI is configured, the service calls Chat Completions with a concise system prompt and returns a reply draft.

# How to Run

* Create a virtual environment and install `requirements.txt`.
* (Optional) Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want `/reply`.
* Start with `uvicorn app.main:app --reload`.
* Open `http://localhost:8000/docs` and exercise the endpoints.
* First call `POST /train_synth` (or `POST /train` with your CSV). Then use `POST /predict`.

# Assumptions & Limits

* Classifier is a simple baseline (TF-IDF + Logistic Regression). It’s fast, explainable, and good for prototypes, but not SOTA.
* Priority scoring is heuristic; consider augmenting with a learned model if you need higher recall/precision.
* OpenAI usage is optional and isolated; the app works without it. Keys are read from environment—**never hard-code secrets**.
* CORS is permissive (`*`) for demo convenience; lock down in production.

# Extensibility Ideas

* Add `/metrics` to report stored test accuracy and label distribution.
* Add `/explain` to show top TF-IDF terms by class for interpretability.
* Support more labels or hierarchical routing.
* Replace/augment the classifier with a transformer-based model; keep the same HTTP contract.

---


## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optionally set OPENAI_API_KEY for /reply


