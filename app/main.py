from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db import Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import app.models  # register tables
        with engine.begin() as conn:
            Base.metadata.create_all(bind=conn)
        logging.info("Schema ensured")
    except Exception as e:
        logging.exception("Schema init failed: %s", e)
    yield

app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers.store_readonly import router as store_readonly_router
from app.routers.intel import router as intel_router
app.include_router(store_readonly_router)
app.include_router(intel_router)

@app.get("/health")
def health():
    return {"status": "ok"}
