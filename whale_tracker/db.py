from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from whale_tracker.models import MonitoredTrader, Position, PositionEvent, TraderProfile


class Database:
    def __init__(self, path: str):
        self.path = str(Path(path))
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traders (
                    address TEXT PRIMARY KEY,
                    first_seen INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL,
                    account_value REAL,
                    total_pnl REAL,
                    pnl_30d REAL,
                    win_rate REAL,
                    avg_trade_duration REAL,
                    avg_position_size REAL,
                    num_trades_30d INTEGER,
                    max_drawdown REAL,
                    sharpe_estimate REAL,
                    score REAL,
                    is_copiable INTEGER DEFAULT 0,
                    is_monitored INTEGER DEFAULT 0,
                    favorite_coins TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_address TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL,
                    leverage REAL,
                    position_value REAL,
                    margin_used REAL,
                    unrealized_pnl REAL,
                    first_seen INTEGER NOT NULL,
                    last_seen INTEGER NOT NULL,
                    closed_at INTEGER,
                    exit_price REAL,
                    realized_pnl REAL,
                    alerted INTEGER DEFAULT 0,
                    FOREIGN KEY (trader_address) REFERENCES traders(address)
                );

                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_address TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    size REAL,
                    unrealized_pnl REAL,
                    mark_price REAL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_address TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    direction TEXT,
                    old_size REAL,
                    new_size REAL,
                    position_value REAL,
                    entry_price REAL,
                    exit_price REAL,
                    realized_pnl REAL,
                    timestamp INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts_sent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_address TEXT NOT NULL,
                    sent_at INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    summary TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(trader_address, closed_at);
                CREATE INDEX IF NOT EXISTS idx_positions_coin ON positions(coin, first_seen);
                CREATE INDEX IF NOT EXISTS idx_snapshots_time ON position_snapshots(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_traders_monitored ON traders(is_monitored, score DESC);
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts_sent(sent_at DESC);
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_trader(self, profile: TraderProfile) -> None:
        now = int(time.time())
        with self._lock:
            cur = self._conn.execute(
                "SELECT first_seen, is_monitored FROM traders WHERE address = ?",
                (profile.address,),
            )
            row = cur.fetchone()
            first_seen = row["first_seen"] if row else now
            is_monitored = row["is_monitored"] if row else 0

            self._conn.execute(
                """
                INSERT INTO traders (
                    address, first_seen, last_updated, account_value, total_pnl,
                    pnl_30d, win_rate, avg_trade_duration, avg_position_size,
                    num_trades_30d, max_drawdown, sharpe_estimate, score,
                    is_copiable, is_monitored, favorite_coins
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    last_updated=excluded.last_updated,
                    account_value=excluded.account_value,
                    total_pnl=excluded.total_pnl,
                    pnl_30d=excluded.pnl_30d,
                    win_rate=excluded.win_rate,
                    avg_trade_duration=excluded.avg_trade_duration,
                    avg_position_size=excluded.avg_position_size,
                    num_trades_30d=excluded.num_trades_30d,
                    max_drawdown=excluded.max_drawdown,
                    sharpe_estimate=excluded.sharpe_estimate,
                    score=excluded.score,
                    is_copiable=excluded.is_copiable,
                    favorite_coins=excluded.favorite_coins
                """,
                (
                    profile.address,
                    first_seen,
                    now,
                    profile.account_value,
                    profile.total_pnl,
                    profile.pnl_30d,
                    profile.win_rate,
                    profile.avg_trade_duration,
                    profile.avg_position_size,
                    profile.num_trades_30d,
                    profile.max_drawdown,
                    profile.sharpe_estimate,
                    profile.score,
                    1 if profile.is_copiable else 0,
                    is_monitored,
                    json.dumps(profile.favorite_coins),
                ),
            )
            self._conn.commit()

    def set_monitored(self, address: str, monitored: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE traders SET is_monitored = ? WHERE address = ?",
                (1 if monitored else 0, address),
            )
            self._conn.commit()

    def set_monitored_bulk(self, addresses: list[str]) -> None:
        with self._lock:
            self._conn.execute("UPDATE traders SET is_monitored = 0")
            if addresses:
                placeholders = ",".join("?" for _ in addresses)
                self._conn.execute(
                    f"UPDATE traders SET is_monitored = 1 WHERE address IN ({placeholders})",
                    addresses,
                )
            self._conn.commit()

    def count_monitored_traders(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM traders WHERE is_monitored = 1"
            )
            return int(cur.fetchone()["n"])

    def get_monitored_traders(self, limit: int = 25) -> list[MonitoredTrader]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT address, score, win_rate, pnl_30d, account_value, total_pnl
                FROM traders
                WHERE is_monitored = 1
                ORDER BY score DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

        return [
            MonitoredTrader(
                address=row["address"],
                score=float(row["score"] or 0.0),
                win_rate=float(row["win_rate"] or 0.0),
                pnl_30d=float(row["pnl_30d"] or 0.0),
                account_value=float(row["account_value"] or 0.0),
                total_pnl=float(row["total_pnl"] or 0.0),
            )
            for row in rows
        ]

    def get_trader_rank(self, address: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT COUNT(*) + 1 AS rank
                FROM traders
                WHERE is_monitored = 1
                  AND score > COALESCE((SELECT score FROM traders WHERE address = ?), 0)
                """,
                (address,),
            )
            row = cur.fetchone()
            return int(row["rank"] if row else 0)

    def get_trader(self, address: str) -> MonitoredTrader | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT address, score, win_rate, pnl_30d, account_value, total_pnl
                FROM traders
                WHERE address = ?
                """,
                (address,),
            )
            row = cur.fetchone()

        if not row:
            return None
        return MonitoredTrader(
            address=row["address"],
            score=float(row["score"] or 0.0),
            win_rate=float(row["win_rate"] or 0.0),
            pnl_30d=float(row["pnl_30d"] or 0.0),
            account_value=float(row["account_value"] or 0.0),
            total_pnl=float(row["total_pnl"] or 0.0),
        )

    def list_all_traders(self, limit: int = 200) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT address, is_monitored, score, win_rate, pnl_30d, account_value, total_pnl
                FROM traders
                ORDER BY score DESC, last_updated DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cur.fetchall()

    def record_position_snapshot(self, trader_address: str, pos: Position, ts_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO position_snapshots (
                    trader_address, coin, timestamp, size, unrealized_pnl, mark_price
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trader_address,
                    pos.coin,
                    ts_ms,
                    pos.size,
                    pos.unrealized_pnl,
                    pos.mark_price,
                ),
            )
            self._conn.commit()

    def upsert_active_position(self, trader_address: str, pos: Position, ts_ms: int) -> None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, first_seen
                FROM positions
                WHERE trader_address = ? AND coin = ? AND closed_at IS NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (trader_address, pos.coin),
            )
            row = cur.fetchone()

            if row:
                self._conn.execute(
                    """
                    UPDATE positions
                    SET direction = ?, size = ?, entry_price = ?, leverage = ?,
                        position_value = ?, margin_used = ?, unrealized_pnl = ?,
                        last_seen = ?
                    WHERE id = ?
                    """,
                    (
                        pos.direction,
                        pos.size,
                        pos.entry_price,
                        pos.leverage,
                        pos.position_value,
                        pos.margin_used,
                        pos.unrealized_pnl,
                        ts_ms,
                        row["id"],
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO positions (
                        trader_address, coin, direction, size, entry_price, leverage,
                        position_value, margin_used, unrealized_pnl,
                        first_seen, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trader_address,
                        pos.coin,
                        pos.direction,
                        pos.size,
                        pos.entry_price,
                        pos.leverage,
                        pos.position_value,
                        pos.margin_used,
                        pos.unrealized_pnl,
                        ts_ms,
                        ts_ms,
                    ),
                )
            self._conn.commit()

    def close_active_position(
        self,
        trader_address: str,
        coin: str,
        ts_ms: int,
        exit_price: float = 0.0,
        realized_pnl: float | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE positions
                SET closed_at = ?, last_seen = ?, exit_price = ?, realized_pnl = ?
                WHERE trader_address = ? AND coin = ? AND closed_at IS NULL
                """,
                (ts_ms, ts_ms, exit_price, realized_pnl, trader_address, coin),
            )
            self._conn.commit()

    def record_event(self, event: PositionEvent) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events (
                    trader_address, coin, event_type, direction,
                    old_size, new_size, position_value,
                    entry_price, exit_price, realized_pnl, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.trader_address,
                    event.coin,
                    event.type,
                    event.direction,
                    event.old_size,
                    event.new_size,
                    event.position_value,
                    event.entry_price,
                    event.exit_price,
                    event.realized_pnl,
                    event.timestamp_ms,
                ),
            )

            if event.type == "CLOSE":
                self._conn.execute(
                    """
                    UPDATE positions
                    SET closed_at = ?, last_seen = ?, exit_price = ?, realized_pnl = ?
                    WHERE trader_address = ? AND coin = ? AND closed_at IS NULL
                    """,
                    (
                        event.timestamp_ms,
                        event.timestamp_ms,
                        event.exit_price,
                        event.realized_pnl,
                        event.trader_address,
                        event.coin,
                    ),
                )
            self._conn.commit()

    def add_alert_log(self, trader_address: str, sent_at: int, event_count: int, summary: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO alerts_sent (trader_address, sent_at, event_count, summary)
                VALUES (?, ?, ?, ?)
                """,
                (trader_address, sent_at, event_count, summary),
            )
            self._conn.commit()

    def count_alerts_since(self, since_ts: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM alerts_sent WHERE sent_at >= ?", (since_ts,)
            )
            return int(cur.fetchone()["n"])

    def last_alert_for_trader(self, trader_address: str) -> int | None:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT sent_at
                FROM alerts_sent
                WHERE trader_address = ?
                ORDER BY sent_at DESC
                LIMIT 1
                """,
                (trader_address,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return int(row["sent_at"])

    def stats(self) -> dict[str, float]:
        with self._lock:
            monitored = self._conn.execute(
                "SELECT COUNT(*) AS n FROM traders WHERE is_monitored = 1"
            ).fetchone()["n"]
            traders = self._conn.execute("SELECT COUNT(*) AS n FROM traders").fetchone()["n"]
            active_positions = self._conn.execute(
                "SELECT COUNT(*) AS n FROM positions WHERE closed_at IS NULL"
            ).fetchone()["n"]
            events_24h = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE timestamp >= ?",
                (int(time.time() * 1000) - 24 * 3600 * 1000,),
            ).fetchone()["n"]
            alerts_24h = self._conn.execute(
                "SELECT COUNT(*) AS n FROM alerts_sent WHERE sent_at >= ?",
                (int(time.time()) - 24 * 3600,),
            ).fetchone()["n"]

        return {
            "traders_total": int(traders),
            "traders_monitored": int(monitored),
            "positions_active": int(active_positions),
            "events_24h": int(events_24h),
            "alerts_24h": int(alerts_24h),
        }
