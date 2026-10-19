"""``python -m server`` starts the API."""

from __future__ import annotations

import uvicorn

from .config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
