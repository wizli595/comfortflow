"""FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from comfortflow.api.routes.comfort import router as comfort_router
from comfortflow.api.routes.models import router as models_router

app = FastAPI(title="ComfortFlow API", version="0.1.0")

app.include_router(comfort_router)
app.include_router(models_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
