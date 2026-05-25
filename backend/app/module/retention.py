"""Background retention worker: prunes Qdrant points and log files past their TTL.

Runs as a single asyncio task spawned by lifecycle.startup(). On each tick:
  1. Sweep Qdrant payloads whose `uploaded_at` is older than DOCUMENT_TTL_DAYS, delete them.
  2. Once per day, delete log files older than LOG_RETENTION_DAYS.

The next-cleanup timestamp is exposed via state() so /gdpr/policy can show it.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Optional

from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


_DATE_FORMAT = '%Y-%m-%d %H:%M'


class RetentionWorker:
    def __init__(self, lifecycle_obj, settings_obj):
        self.lifecycle = lifecycle_obj
        self.settings = settings_obj
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self.last_run_at: Optional[float] = None
        self.next_run_at: Optional[float] = None
        self.last_deleted_points: int = 0
        self.last_deleted_logs: int = 0
        self._log_sweep_done_for_day: Optional[str] = None

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())
        logger.info("RetentionWorker started (doc_ttl=%s days, log_retention=%s days, interval=%ss)",
                    self.settings.DOCUMENT_TTL_DAYS, self.settings.LOG_RETENTION_DAYS,
                    self.settings.RETENTION_SCAN_INTERVAL_SECONDS)

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run(self) -> None:
        interval = max(60, self.settings.RETENTION_SCAN_INTERVAL_SECONDS)
        while not (self._stop_event and self._stop_event.is_set()):
            try:
                self.last_run_at = time.time()
                self.next_run_at = self.last_run_at + interval
                await self._sweep_documents()
                await self._sweep_logs_once_per_day()
            except Exception as e:
                logger.exception("RetentionWorker tick failed: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)  # type: ignore[arg-type]
                break  # stop_event was set
            except asyncio.TimeoutError:
                continue

    async def _sweep_documents(self) -> None:
        client = getattr(self.lifecycle, 'qdrant_client', None)
        if client is None:
            return
        collection = self.settings.qdrant_collection
        ttl_days = float(self.settings.DOCUMENT_TTL_DAYS)
        if ttl_days <= 0:
            return

        cutoff = datetime.datetime.now() - datetime.timedelta(days=ttl_days)

        ids_to_delete = []
        offset = None
        try:
            while True:
                result = await asyncio.to_thread(
                    client.scroll,
                    collection_name=collection,
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = result if isinstance(result, tuple) else (result, None)
                if not points:
                    break
                for p in points:
                    payload = getattr(p, 'payload', {}) or {}
                    uploaded_at = payload.get('uploaded_at') or ''
                    try:
                        ts = datetime.datetime.strptime(uploaded_at, _DATE_FORMAT)
                    except Exception:
                        continue
                    if ts < cutoff:
                        ids_to_delete.append(p.id)
                if not next_offset:
                    break
                offset = next_offset
        except Exception as e:
            logger.exception("Retention scroll failed: %s", e)
            return

        if not ids_to_delete:
            self.last_deleted_points = 0
            return

        try:
            await asyncio.to_thread(
                client.delete,
                collection_name=collection,
                points_selector=qmodels.PointIdsList(points=ids_to_delete),
            )
            self.last_deleted_points = len(ids_to_delete)
            logger.info("Retention: deleted %d expired Qdrant points (older than %.3f days)",
                        len(ids_to_delete), ttl_days)
        except Exception as e:
            logger.exception("Retention delete failed: %s", e)

    async def _sweep_logs_once_per_day(self) -> None:
        today = datetime.date.today().isoformat()
        if self._log_sweep_done_for_day == today:
            return

        log_dir = Path(__file__).resolve().parent.parent.parent / 'logs'
        if not log_dir.exists():
            self._log_sweep_done_for_day = today
            return

        cutoff_ts = time.time() - (self.settings.LOG_RETENTION_DAYS * 86400)
        deleted = 0
        for f in log_dir.glob('*.log*'):
            try:
                if f.stat().st_mtime < cutoff_ts:
                    f.unlink()
                    deleted += 1
            except Exception:
                continue
        self.last_deleted_logs = deleted
        self._log_sweep_done_for_day = today
        if deleted:
            logger.info("Retention: deleted %d log file(s) older than %d days",
                        deleted, self.settings.LOG_RETENTION_DAYS)

    def state(self) -> dict:
        return {
            'document_ttl_days': self.settings.DOCUMENT_TTL_DAYS,
            'log_retention_days': self.settings.LOG_RETENTION_DAYS,
            'scan_interval_seconds': self.settings.RETENTION_SCAN_INTERVAL_SECONDS,
            'last_run_at': datetime.datetime.fromtimestamp(self.last_run_at).isoformat() if self.last_run_at else None,
            'next_run_at': datetime.datetime.fromtimestamp(self.next_run_at).isoformat() if self.next_run_at else None,
            'last_deleted_points': self.last_deleted_points,
            'last_deleted_logs': self.last_deleted_logs,
        }


_worker: Optional[RetentionWorker] = None


def get_worker(lifecycle_obj, settings_obj) -> RetentionWorker:
    global _worker
    if _worker is None:
        _worker = RetentionWorker(lifecycle_obj, settings_obj)
    return _worker
