"""API error definitions and exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Structured API error raised across Studio endpoints."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 501,
        issues: list[dict[str, Any]] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status = status
        self.issues = issues or []
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI application."""

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "issues": exc.issues,
                }
            },
        )
