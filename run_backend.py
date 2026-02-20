#!/usr/bin/env python3
"""Run FastAPI backend for backtesting terminal."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()

