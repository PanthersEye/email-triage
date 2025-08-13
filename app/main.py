from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import classifier

tags_metadata = [
    {"name": "Email Triage", "description": "Training, prediction, priority scoring, and reply drafting."}
]

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Classify inbound emails (support/sales/hr/finance/general), "
        "estimate priority, and optionally draft replies."
    ),
    contact={"name": "Triage Team", "email": "triage@example.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(classifier.router)
