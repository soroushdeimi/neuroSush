"""Serve the schema, spec validation and the built Compose interface."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from studio.api import health, probe, projects, runs, schema, stream, validate
from studio.config import config
from studio.errors import register_exception_handlers

app = FastAPI(title="neuroSush Studio", version="0.1.0")

register_exception_handlers(app)

app.include_router(health.router, prefix="/api")
app.include_router(schema.router, prefix="/api")
app.include_router(validate.router, prefix="/api")
app.include_router(probe.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(stream.router, prefix="/api")

if config.web_root.is_dir():
    app.mount("/", StaticFiles(directory=config.web_root, html=True), name="web")
