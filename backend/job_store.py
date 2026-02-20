from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str
    payload: dict[str, Any]
    created_at: str
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    result: dict[str, Any] | None = None
    progress: float = 0.0
    message: str = ""


class JobStore:
    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bt-job")

    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        runner: Callable[[Callable[[float, str], None]], dict[str, Any]],
    ) -> JobRecord:
        job_id = str(uuid4())
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            status="queued",
            payload=payload,
            created_at=utc_now_iso(),
            message="queued",
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run, job_id, runner)
        return record

    def _run(
        self,
        job_id: str,
        runner: Callable[[Callable[[float, str], None]], dict[str, Any]],
    ) -> None:
        def report(progress: float, message: str) -> None:
            with self._lock:
                rec = self._jobs.get(job_id)
                if not rec:
                    return
                rec.progress = max(0.0, min(1.0, float(progress)))
                rec.message = str(message)

        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = "running"
            rec.started_at = utc_now_iso()
            rec.message = "running"

        try:
            result = runner(report)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                rec = self._jobs.get(job_id)
                if not rec:
                    return
                rec.status = "failed"
                rec.finished_at = utc_now_iso()
                rec.error = str(exc)
                rec.message = "failed"
                rec.progress = 1.0
            return

        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = "completed"
            rec.finished_at = utc_now_iso()
            rec.result = result
            rec.progress = 1.0
            rec.message = "completed"

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        with self._lock:
            rows = list(self._jobs.values())
        rows.sort(key=lambda x: x.created_at, reverse=True)
        return rows
