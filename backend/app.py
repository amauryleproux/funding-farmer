from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.job_store import JobRecord, JobStore
from backend.services import (
    ALLOWED_TIMEFRAMES,
    DEFAULT_DB,
    describe_timeframes,
    export_blotter,
    export_equity_curve,
    get_strategy_catalog,
    list_symbols,
    run_ai_backtests,
    run_compare_backtests,
    run_single_backtest,
)


class BacktestRequest(BaseModel):
    strategy_id: str = Field(..., description="Identifiant de strategie")
    timeframe: str = Field(default="1h")
    start: str = Field(default="", description="YYYY-MM-DD")
    end: str = Field(default="", description="YYYY-MM-DD")
    db: str = Field(default=DEFAULT_DB)
    max_tokens: int = Field(default=25, ge=1, le=200)
    min_candles: int = Field(default=200, ge=50, le=5000)
    overrides: dict[str, Any] = Field(default_factory=dict)
    run_async: bool = Field(default=True)


class CompareRequest(BaseModel):
    strategy_ids: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=lambda: list(ALLOWED_TIMEFRAMES))
    window_mode: str = Field(default="both")
    start: str = Field(default="", description="YYYY-MM-DD")
    end: str = Field(default="", description="YYYY-MM-DD")
    db: str = Field(default=DEFAULT_DB)
    max_tokens: int = Field(default=25, ge=1, le=200)
    min_candles: int = Field(default=200, ge=50, le=5000)
    overrides: dict[str, Any] = Field(default_factory=dict)
    run_async: bool = Field(default=True)


class AIBacktestRequest(BaseModel):
    strategy_ids: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=lambda: list(ALLOWED_TIMEFRAMES))
    start: str = Field(default="", description="YYYY-MM-DD")
    end: str = Field(default="", description="YYYY-MM-DD")
    db: str = Field(default=DEFAULT_DB)
    max_tokens: int = Field(default=25, ge=1, le=200)
    min_candles: int = Field(default=200, ge=50, le=5000)
    max_runs: int = Field(default=120, ge=10, le=2000)
    top_n: int = Field(default=5, ge=1, le=20)
    min_trades: int = Field(default=5, ge=0, le=1000)
    max_drawdown_pct: float = Field(default=25.0, gt=0, le=100.0)
    objective: str = Field(default="balanced")
    seed: int = Field(default=42)
    force_refresh: bool = Field(default=False)
    overrides: dict[str, Any] = Field(default_factory=dict)
    run_async: bool = Field(default=True)


app = FastAPI(
    title="Funding Farmer Backtest Terminal API",
    version="1.0.0",
    description=(
        "Backend de backtest orienté Bloomberg-terminal lookalike: catalogue stratégies, "
        "jobs async, blotter, equity curve et leaderboard."
    ),
)
jobs = JobStore(max_workers=2)
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/terminal-static", StaticFiles(directory=str(STATIC_DIR)), name="terminal-static")


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # pydantic v2
    return model.dict()  # pydantic v1 fallback


def _job_payload(record: JobRecord, include_result: bool = False) -> dict[str, Any]:
    payload = {
        "job_id": record.job_id,
        "kind": record.kind,
        "status": record.status,
        "progress": record.progress,
        "message": record.message,
        "created_at": record.created_at,
        "started_at": record.started_at or None,
        "finished_at": record.finished_at or None,
        "error": record.error or None,
    }
    if include_result and record.result:
        if record.kind == "single":
            payload["summary"] = record.result.get("summary")
        elif record.kind == "compare":
            payload["summary"] = record.result.get("summary")
            payload["leaderboard_top"] = record.result.get("leaderboard", [])[:10]
        elif record.kind == "ai_optimize":
            payload["summary"] = record.result.get("summary")
            payload["top_picks"] = record.result.get("top_picks", [])[:5]
    return payload


def _require_job(job_id: str) -> JobRecord:
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job introuvable: {job_id}")
    return record


def _require_completed(job_id: str) -> JobRecord:
    record = _require_job(job_id)
    if record.status == "failed":
        raise HTTPException(status_code=400, detail=record.error or "Job failed")
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job non termine (status={record.status})")
    if not record.result:
        raise HTTPException(status_code=500, detail="Resultat manquant")
    return record


@app.get("/")
def root() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Dashboard statique indisponible")
    return FileResponse(index)


@app.get("/terminal")
def terminal() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Dashboard statique indisponible")
    return FileResponse(index)


@app.get("/api/v1/status")
def status() -> dict[str, Any]:
    return {
        "service": "Funding Farmer Backtest Terminal API",
        "mode": "bloomberg_terminal_lookalike_backend",
        "version": "1.0.0",
        "routes": [
            "/",
            "/terminal",
            "/api/v1/status",
            "/api/v1/strategies",
            "/api/v1/timeframes",
            "/api/v1/symbols",
            "/api/v1/backtests",
            "/api/v1/backtests/compare",
            "/api/v1/backtests/ai-run",
            "/api/v1/jobs",
        ],
    }


@app.get("/api/v1/strategies")
def strategies() -> dict[str, Any]:
    rows = get_strategy_catalog()
    return {"count": len(rows), "items": rows}


@app.get("/api/v1/timeframes")
def timeframes(db: str = Query(default=DEFAULT_DB)) -> dict[str, Any]:
    try:
        return describe_timeframes(db=db)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/symbols")
def symbols(
    timeframe: str = Query(default="1h"),
    db: str = Query(default=DEFAULT_DB),
    limit: int = Query(default=30, ge=1, le=500),
) -> dict[str, Any]:
    try:
        rows = list_symbols(db=db, timeframe=timeframe, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(rows), "items": rows}


@app.post("/api/v1/backtests")
def create_backtest(req: BacktestRequest) -> dict[str, Any]:
    payload = _model_to_dict(req)
    if req.run_async:
        record = jobs.submit(
            kind="single",
            payload=payload,
            runner=lambda report: run_single_backtest(payload, report=report),
        )
        return _job_payload(record, include_result=False)

    try:
        result = run_single_backtest(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "completed", "kind": "single", "summary": result.get("summary")}


@app.post("/api/v1/backtests/compare")
def create_compare(req: CompareRequest) -> dict[str, Any]:
    payload = _model_to_dict(req)
    if req.run_async:
        record = jobs.submit(
            kind="compare",
            payload=payload,
            runner=lambda report: run_compare_backtests(payload, report=report),
        )
        return _job_payload(record, include_result=False)

    try:
        result = run_compare_backtests(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "completed",
        "kind": "compare",
        "summary": result.get("summary"),
        "leaderboard_top": result.get("leaderboard", [])[:20],
    }


@app.post("/api/v1/backtests/ai-run")
def create_ai_run(req: AIBacktestRequest) -> dict[str, Any]:
    payload = _model_to_dict(req)
    if req.run_async:
        record = jobs.submit(
            kind="ai_optimize",
            payload=payload,
            runner=lambda report: run_ai_backtests(payload, report=report),
        )
        return _job_payload(record, include_result=False)

    try:
        result = run_ai_backtests(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "completed",
        "kind": "ai_optimize",
        "summary": result.get("summary"),
        "top_picks": result.get("top_picks", [])[: req.top_n],
    }


@app.get("/api/v1/jobs")
def list_jobs() -> dict[str, Any]:
    rows = [_job_payload(job, include_result=False) for job in jobs.list()]
    return {"count": len(rows), "items": rows}


@app.get("/api/v1/backtests/{job_id}")
def backtest_status(job_id: str) -> dict[str, Any]:
    record = _require_job(job_id)
    return _job_payload(record, include_result=True)


@app.get("/api/v1/backtests/{job_id}/summary")
def backtest_summary(job_id: str) -> dict[str, Any]:
    record = _require_completed(job_id)
    if record.kind != "single":
        raise HTTPException(status_code=400, detail="Endpoint reserve aux jobs single")
    return {"job_id": job_id, "summary": record.result["summary"]}


@app.get("/api/v1/backtests/{job_id}/blotter")
def backtest_blotter(
    job_id: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    record = _require_completed(job_id)
    if record.kind != "single":
        raise HTTPException(status_code=400, detail="Endpoint reserve aux jobs single")
    try:
        data = export_blotter(record.result, limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, **data}


@app.get("/api/v1/backtests/{job_id}/equity")
def backtest_equity(
    job_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    record = _require_completed(job_id)
    if record.kind != "single":
        raise HTTPException(status_code=400, detail="Endpoint reserve aux jobs single")
    try:
        data = export_equity_curve(record.result, limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, **data}


@app.get("/api/v1/backtests/{job_id}/leaderboard")
def compare_leaderboard(job_id: str, top: int = Query(default=20, ge=1, le=500)) -> dict[str, Any]:
    record = _require_completed(job_id)
    if record.kind not in {"compare", "ai_optimize"}:
        raise HTTPException(status_code=400, detail="Endpoint reserve aux jobs compare/ai")
    leaderboard = record.result.get("leaderboard", [])
    return {"job_id": job_id, "count": len(leaderboard), "items": leaderboard[:top]}


@app.get("/api/v1/backtests/{job_id}/matrix")
def compare_matrix(job_id: str, top: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    record = _require_completed(job_id)
    if record.kind != "compare":
        raise HTTPException(status_code=400, detail="Endpoint reserve aux jobs compare")
    rows = record.result.get("rows", [])
    return {"job_id": job_id, "count": len(rows), "items": rows[:top]}


@app.get("/api/v1/backtests/{job_id}/ai-top")
def ai_top(job_id: str, top: int = Query(default=5, ge=1, le=500)) -> dict[str, Any]:
    record = _require_completed(job_id)
    if record.kind != "ai_optimize":
        raise HTTPException(status_code=400, detail="Endpoint reserve aux jobs ai_optimize")
    picks = record.result.get("top_picks", [])
    return {"job_id": job_id, "count": len(picks), "items": picks[:top], "summary": record.result.get("summary")}
