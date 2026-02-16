#!/usr/bin/env python3
"""
=============================================================================
 HYPERPULSE — See the squeeze before it fires.
=============================================================================
 Telegram alert bot for Hyperliquid squeeze setups.

 Scans all Hyperliquid perp pairs, detects squeezes,
 predicts direction, adjusts with funding rates,
 and sends formatted Telegram alerts.

 Usage:
   # First launch (collects ~200 candles per token, ~10 min)
   python hyperpulse_bot.py --telegram-token YOUR_BOT_TOKEN --channel-id @your_channel

   # Dry-run mode (prints to terminal, no Telegram)
   python hyperpulse_bot.py --dry-run

   # Custom config
   python hyperpulse_bot.py --telegram-token TOKEN --channel-id @chan \
       --scan-interval 300 --min-score 0.80 --min-confidence 0.85

 Requirements:
   pip install requests pandas numpy python-telegram-bot --break-system-packages

 Structure:
   hyperpulse_bot.py      <- Main bot
   squeeze_detector.py    <- Squeeze detection engine
   trend_filter.py        <- BTC + token trend filter
   structure_analysis.py  <- Market structure analysis
   hyperpulse.db          <- SQLite database (auto-created)
=============================================================================
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from squeeze_detector import (
    SqueezeDetector,
    SqueezeConfig,
    SqueezeSignal,
    SqueezePhase,
    BreakoutDirection,
)
from trend_filter import compute_trend, should_block_signal
from structure_analysis import analyze_structure, should_take_signal


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hyperpulse")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class HyperPulseConfig:
    """HyperPulse bot configuration."""

    # --- Telegram ---
    telegram_token: str = ""
    channel_id: str = ""            # @channel_name or -100xxxxx (chat ID)
    premium_channel_id: str = ""    # Premium channel (optional)

    # --- Mode ---
    dry_run: bool = False           # No Telegram, print only

    # --- Scan ---
    scan_interval_sec: int = 300    # 5 minutes between scans
    data_refresh_sec: int = 900     # 15 min between full candle refresh
    warmup_candles: int = 200       # Number of candles for indicator warm-up

    # --- Signal filters ---
    min_squeeze_score: float = 0.80
    min_direction_confidence: float = 0.85
    min_expected_move_pct: float = 0.02
    min_ttm_squeeze_bars: int = 3   # Minimum TTM squeeze bars
    required_phases: list = field(
        default_factory=lambda: ["ready", "firing"]
    )
    # Tokens: min/max volume filter
    min_volume_24h: float = 100_000
    max_volume_24h: float = 500_000_000

    # --- Funding rate adjustment ---
    funding_boost_max: float = 0.15     # +/-15% confidence adjustment
    funding_strong_threshold: float = 0.0003  # 0.03%/h = strong signal

    # --- Signal tracking ---
    signal_ttl_hours: int = 24          # Resolve signals after 24h
    cooldown_per_token_min: int = 120   # Min 2h between alerts for same token

    # --- Scheduled messages ---
    daily_summary_hour_utc: int = 21    # 21h UTC
    morning_briefing_hour_utc: int = 8  # 8h UTC
    send_resolution_alerts: bool = True # Send alert when signal is resolved

    # --- Free tier limits ---
    free_max_alerts_per_day: int = 5
    free_top_n_tokens: int = 15         # Top 15 by volume only

    # --- Database ---
    db_path: str = "hyperpulse.db"

    # --- Hyperliquid API ---
    hl_api_url: str = "https://api.hyperliquid.xyz/info"


# =============================================================================
# DATABASE
# =============================================================================

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize the SQLite database for HyperPulse."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            coin            TEXT NOT NULL,
            timestamp       INTEGER NOT NULL,
            phase           TEXT NOT NULL,
            direction       TEXT NOT NULL,
            score           REAL NOT NULL,
            confidence      REAL NOT NULL,
            funding_rate    REAL DEFAULT 0,
            funding_aligned INTEGER DEFAULT 0,
            entry_price     REAL NOT NULL,
            target_price    REAL,
            stop_price      REAL,
            expected_move   REAL,
            -- Resolution
            resolved        INTEGER DEFAULT 0,
            resolved_at     INTEGER,
            exit_price      REAL,
            result          TEXT,           -- 'win', 'loss', 'expired'
            pnl_pct         REAL,
            -- Meta
            bb_width_pct    REAL,
            atr_value       REAL,
            volume_ratio    REAL,
            alerted         INTEGER DEFAULT 0,
            created_at      INTEGER DEFAULT (strftime('%s','now'))
        );

        CREATE TABLE IF NOT EXISTS daily_stats (
            date            TEXT PRIMARY KEY,
            signals_total   INTEGER DEFAULT 0,
            signals_long    INTEGER DEFAULT 0,
            signals_short   INTEGER DEFAULT 0,
            wins            INTEGER DEFAULT 0,
            losses          INTEGER DEFAULT 0,
            expired         INTEGER DEFAULT 0,
            avg_score       REAL DEFAULT 0,
            avg_confidence  REAL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_signals_active
            ON signals(resolved, coin, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_signals_coin
            ON signals(coin, created_at DESC);
    """)
    conn.commit()
    return conn


# =============================================================================
# HYPERLIQUID DATA FETCHER (standalone, no external dependency)
# =============================================================================

class HyperliquidData:
    """Fetches OHLCV data and funding rates from Hyperliquid."""

    def __init__(self, api_url: str = "https://api.hyperliquid.xyz/info"):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def _post(self, payload: dict, timeout: int = 30) -> dict | list:
        resp = self.session.post(self.api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_all_tokens(
        self, min_vol: float = 100_000, max_vol: float = 500_000_000
    ) -> list[dict]:
        """Get all perp tokens with volume, funding, OI."""
        ctx_resp = self._post({"type": "metaAndAssetCtxs"})
        meta = ctx_resp[0] if isinstance(ctx_resp, list) else {}
        contexts = ctx_resp[1] if isinstance(ctx_resp, list) and len(ctx_resp) > 1 else []
        universe = meta.get("universe", [])

        tokens = []
        for i, asset in enumerate(universe):
            name = asset.get("name", "")
            if i >= len(contexts):
                continue
            ctx = contexts[i]
            vol = float(ctx.get("dayNtlVlm", 0))
            if vol < min_vol or vol > max_vol:
                continue

            tokens.append({
                "symbol": name,
                "volume_24h": vol,
                "funding_rate": float(ctx.get("funding", 0)),
                "mark_price": float(ctx.get("markPx", 0)),
                "open_interest": float(ctx.get("openInterest", 0)) * float(ctx.get("markPx", 1)),
            })

        tokens.sort(key=lambda t: t["volume_24h"], reverse=True)
        return tokens

    def fetch_candles(
        self, coin: str, interval: str = "1h", limit: int = 200
    ) -> pd.DataFrame:
        """Fetch OHLCV candles as DataFrame ready for SqueezeDetector."""
        interval_ms_map = {
            "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "4h": 14_400_000,
        }
        interval_ms = interval_ms_map.get(interval, 3_600_000)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (limit * interval_ms)

        raw = self._post({
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": now_ms,
            },
        })

        if not isinstance(raw, list) or len(raw) == 0:
            return pd.DataFrame()

        rows = []
        for c in raw:
            rows.append({
                "t": int(c.get("t", 0)),
                "open": float(c.get("o", 0)),
                "high": float(c.get("h", 0)),
                "low": float(c.get("l", 0)),
                "close": float(c.get("c", 0)),
                "volume": float(c.get("v", 0)),
            })

        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df

    def get_funding_rates(self) -> dict[str, float]:
        """Get current funding rates for all tokens."""
        ctx_resp = self._post({"type": "metaAndAssetCtxs"})
        meta = ctx_resp[0] if isinstance(ctx_resp, list) else {}
        contexts = ctx_resp[1] if isinstance(ctx_resp, list) and len(ctx_resp) > 1 else []
        universe = meta.get("universe", [])

        rates = {}
        for i, asset in enumerate(universe):
            if i < len(contexts):
                rates[asset.get("name", "")] = float(contexts[i].get("funding", 0))
        return rates


# =============================================================================
# FUNDING RATE ADJUSTMENT
# =============================================================================

def compute_funding_adjustment(
    direction: BreakoutDirection,
    funding_rate: float,
    config: HyperPulseConfig,
) -> tuple[float, bool]:
    """
    Compute confidence adjustment based on funding alignment.

    Returns: (confidence_adjustment, is_aligned)

    Logic:
    - Positive funding = market is bullish = confirms LONG
    - Negative funding = market is bearish = confirms SHORT
    """
    if direction == BreakoutDirection.UNKNOWN or funding_rate == 0:
        return 0.0, False

    is_long = direction == BreakoutDirection.LONG

    # Positive funding = bullish sentiment = good for longs
    # Negative funding = bearish sentiment = good for shorts
    if is_long:
        alignment = funding_rate   # Positive funding = bullish = good for longs
    else:
        alignment = -funding_rate  # Negative funding = bearish = good for shorts

    is_aligned = alignment > 0

    # Scale: normalize by strong threshold
    scale = max(-1.0, min(1.0, alignment / config.funding_strong_threshold))

    # Confidence adjustment
    conf_adj = scale * config.funding_boost_max

    return conf_adj, is_aligned


# =============================================================================
# TELEGRAM FORMATTER
# =============================================================================

def format_signal_alert(
    signal: SqueezeSignal,
    funding_rate: float,
    funding_aligned: bool,
    conf_adjusted: float,
    entry_price: float,
    target_price: float,
    stop_price: float,
    track_record: dict,
    structure_info: str = "",
) -> str:
    """Format a squeeze signal as a Telegram message."""

    is_long = signal.direction == BreakoutDirection.LONG
    dir_emoji = "🟢" if is_long else "🔴"
    dir_text = "LONG 📈" if is_long else "SHORT 📉"
    phase_text = signal.phase.value.upper()

    # Funding info
    funding_pct = funding_rate * 100
    if funding_aligned:
        funding_status = "✅ Aligned"
    elif abs(funding_rate) < 0.0001:
        funding_status = "➖ Neutral"
    else:
        funding_status = "⚠️ Against"

    # Target/stop distances
    if is_long:
        target_dist = (target_price - entry_price) / entry_price * 100
        stop_dist = (entry_price - stop_price) / entry_price * 100
    else:
        target_dist = (entry_price - target_price) / entry_price * 100
        stop_dist = (stop_price - entry_price) / entry_price * 100

    # Win rate
    total = track_record.get("total", 0)
    wins = track_record.get("wins", 0)
    wr = (wins / total * 100) if total > 0 else 0

    msg = (
        f"{dir_emoji} *SQUEEZE ALERT — {signal.coin}/USDC*\n"
        f"\n"
        f"📊 Signal: *{dir_text}* (conf {conf_adjusted:.0%})\n"
        f"⚡ Phase: `{phase_text}` | Score: `{signal.score:.2f}`\n"
        f"💰 Funding: `{funding_pct:+.4f}%/h` {funding_status}\n"
        f"💵 Price: `${entry_price:.4f}`\n"
        f"🎯 Target: `${target_price:.4f}` (+{target_dist:.1f}%)\n"
        f"🛑 Stop: `${stop_price:.4f}` (-{stop_dist:.1f}%)\n"
    )

    if structure_info:
        msg += f"\n📐 _{structure_info}_\n"

    msg += (
        f"\n"
        f"_Indicators:_\n"
        f"  ├ BB width: `P{signal.bb_width_percentile:.0f}` "
        f"{'✅' if signal.bb_width_percentile < 20 else '⚠️'}\n"
        f"  ├ TTM Squeeze: "
        f"{'✅ ' + str(signal.ttm_squeeze_bars) + ' bars' if signal.ttm_squeeze else '❌'}\n"
        f"  ├ ATR: `P{signal.atr_percentile:.0f}` "
        f"{'✅' if signal.atr_percentile < 25 else '⚠️'}\n"
        f"  ├ Volume ratio: `{signal.volume_ratio:.1f}x`\n"
        f"  └ Expected move: `{signal.expected_move_pct:.1%}`\n"
        f"\n"
        f"📊 Track record: {wr:.0f}% win rate ({total} signals)"
    )

    return msg


def format_daily_summary(stats: dict, top_signals: list = None) -> str:
    """Format daily summary message."""
    total = stats.get('total', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    expired = stats.get('expired', 0)
    wr = stats.get('win_rate', 0)
    avg_pnl = stats.get('avg_pnl_pct', 0)

    # Header with date
    date_str = datetime.now(timezone.utc).strftime("%d/%m/%Y")

    msg = (
        f"📊 *HyperPulse — Daily Recap {date_str}*\n"
        f"\n"
        f"*Signals emitted:* {total}\n"
        f"  ├ Long: {stats.get('longs', 0)} | Short: {stats.get('shorts', 0)}\n"
        f"  ├ ✅ Wins: {wins}\n"
        f"  ├ ❌ Losses: {losses}\n"
        f"  └ ⏰ Expired: {expired}\n"
    )

    if wins + losses > 0:
        msg += (
            f"\n"
            f"*Performance:*\n"
            f"  ├ Win rate: *{wr:.0f}%*\n"
            f"  ├ Avg PnL: `{avg_pnl:+.2f}%`\n"
            f"  └ Avg score: `{stats.get('avg_score', 0):.2f}`\n"
        )

    # Top signals of the day
    if top_signals:
        msg += f"\n*Top signals:*\n"
        for s in top_signals[:5]:
            emoji = "✅" if "win" in (s.get("result") or "") else "❌" if s.get("result") else "⏳"
            pnl = s.get("pnl_pct", 0)
            msg += f"  {emoji} {s['coin']} {s['direction'].upper()} → `{pnl:+.1%}`\n"

    # All-time stats
    all_time = stats.get("all_time", {})
    if all_time.get("total", 0) > 0:
        msg += (
            f"\n*All-time ({all_time['total']} signals):*\n"
            f"  Win rate: *{all_time.get('win_rate', 0):.0f}%*\n"
        )

    msg += f"\n_hyper-pulse.xyz — See the squeeze before it fires._"
    return msg


def format_resolution_alert(
    coin: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    result: str,
    duration_hours: float,
    score: float,
) -> str:
    """Format a signal resolution alert."""
    is_win = "win" in result
    emoji = "✅" if is_win else "❌"
    dir_emoji = "📈" if direction == "long" else "📉"
    result_text = "TARGET ✅" if result == "win" else "STOP ❌" if result == "loss" else f"EXPIRED {'✅' if is_win else '❌'}"

    msg = (
        f"{emoji} *RESULT — {coin}/USDC*\n"
        f"\n"
        f"{dir_emoji} {direction.upper()} | Score: `{score:.2f}`\n"
        f"💵 Entry: `${entry_price:.4f}` → Exit: `${exit_price:.4f}`\n"
        f"📊 PnL: *{pnl_pct*100:+.1f}%*\n"
        f"⏱ Duration: {duration_hours:.1f}h\n"
        f"🏁 {result_text}"
    )
    return msg


def format_morning_briefing(
    building_signals: list,
    active_signals: list,
    market_stats: dict,
) -> str:
    """Format morning briefing with watchlist."""
    date_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")

    msg = f"☀️ *HyperPulse — Briefing {date_str}*\n"

    # Active (unresolved) signals
    if active_signals:
        msg += f"\n*📍 Open positions ({len(active_signals)}):*\n"
        for s in active_signals[:5]:
            dir_emoji = "📈" if s["direction"] == "long" else "📉"
            pnl = s.get("current_pnl_pct", 0)
            pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            msg += f"  {dir_emoji} {s['coin']} → {pnl_emoji} `{pnl*100:+.1f}%`\n"

    # Building squeezes (watchlist)
    if building_signals:
        msg += f"\n*👀 Building squeezes ({len(building_signals)}):*\n"
        for s in building_signals[:8]:
            ttm = "🔵" if s.get("ttm_squeeze") else "⚪"
            msg += (
                f"  {ttm} {s['coin']} — score `{s['score']:.2f}` "
                f"| BB `P{s.get('bb_pct', 50):.0f}` "
                f"| TTM {s.get('ttm_bars', 0)} bars\n"
            )
    else:
        msg += f"\n_No building squeezes at the moment._\n"

    # Market overview
    if market_stats:
        msg += (
            f"\n*📈 Market:*\n"
            f"  Tokens scanned: {market_stats.get('total_tokens', 0)}\n"
            f"  Avg funding: `{market_stats.get('avg_funding', 0):+.4f}%/h`\n"
        )

    msg += f"\n_hyper-pulse.xyz — See the squeeze before it fires._"
    return msg


# =============================================================================
# TELEGRAM SENDER
# =============================================================================

class TelegramSender:
    """Sends messages to Telegram channel."""

    def __init__(self, token: str, default_channel: str = ""):
        self.token = token
        self.default_channel = default_channel
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()

    def send_message(
        self, text: str, channel_id: str = "", parse_mode: str = "Markdown"
    ) -> bool:
        """Send a message to a Telegram channel/chat."""
        chat_id = channel_id or self.default_channel
        if not chat_id:
            log.warning("No Telegram channel configured")
            return False

        try:
            resp = self.session.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            result = resp.json()
            if not result.get("ok"):
                log.error(f"Telegram error: {result.get('description', 'unknown')}")
                return False
            return True
        except Exception as e:
            log.error(f"Telegram send error: {e}")
            return False


# =============================================================================
# SIGNAL TRACKER
# =============================================================================

class SignalTracker:
    """Tracks signals and auto-resolves them."""

    def __init__(self, conn: sqlite3.Connection, config: HyperPulseConfig):
        self.conn = conn
        self.config = config

    def record_signal(
        self,
        signal: SqueezeSignal,
        entry_price: float,
        target_price: float,
        stop_price: float,
        funding_rate: float,
        funding_aligned: bool,
        confidence_adjusted: float,
    ) -> int:
        """Record a new signal. Returns signal ID."""
        cursor = self.conn.execute(
            """INSERT INTO signals
               (coin, timestamp, phase, direction, score, confidence,
                funding_rate, funding_aligned, entry_price, target_price,
                stop_price, expected_move, bb_width_pct, atr_value, volume_ratio)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.coin,
                int(time.time()),
                signal.phase.value,
                signal.direction.value,
                signal.score,
                confidence_adjusted,
                funding_rate,
                1 if funding_aligned else 0,
                entry_price,
                target_price,
                stop_price,
                signal.expected_move_pct,
                signal.bb_width_percentile,
                signal.atr_value,
                signal.volume_ratio,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def mark_alerted(self, signal_id: int):
        """Mark signal as alerted (sent to Telegram)."""
        self.conn.execute(
            "UPDATE signals SET alerted = 1 WHERE id = ?", (signal_id,)
        )
        self.conn.commit()

    def resolve_signals(self, hl_data: HyperliquidData) -> list[dict]:
        """Auto-resolve open signals. Returns list of newly resolved signals."""
        open_signals = self.conn.execute(
            """SELECT id, coin, direction, entry_price, target_price,
                      stop_price, timestamp, score
               FROM signals WHERE resolved = 0"""
        ).fetchall()

        now = time.time()
        resolved_list = []

        # Get all mids once (not per signal)
        try:
            mids = hl_data._post({"type": "allMids"})
        except:
            return []

        for row in open_signals:
            sid, coin, direction, entry_px, target_px, stop_px, ts, score = row
            age_hours = (now - ts) / 3600

            current_price = float(mids.get(coin, 0))
            if current_price <= 0:
                continue

            is_long = direction == "long"
            result = None
            exit_price = current_price

            # Check target hit
            if is_long and current_price >= target_px:
                result = "win"
            elif not is_long and current_price <= target_px:
                result = "win"
            # Check stop hit
            elif is_long and current_price <= stop_px:
                result = "loss"
            elif not is_long and current_price >= stop_px:
                result = "loss"
            # TTL expired
            elif age_hours >= self.config.signal_ttl_hours:
                if is_long:
                    result = "win" if current_price > entry_px else "loss"
                else:
                    result = "win" if current_price < entry_px else "loss"
                result = f"expired_{result}"

            if result:
                if is_long:
                    pnl_pct = (exit_price - entry_px) / entry_px
                else:
                    pnl_pct = (entry_px - exit_price) / entry_px

                self.conn.execute(
                    """UPDATE signals SET
                        resolved = 1, resolved_at = ?, exit_price = ?,
                        result = ?, pnl_pct = ?
                       WHERE id = ?""",
                    (int(now), exit_price, result, pnl_pct, sid),
                )

                resolved_list.append({
                    "id": sid,
                    "coin": coin,
                    "direction": direction,
                    "entry_price": entry_px,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "result": result,
                    "duration_hours": age_hours,
                    "score": score,
                })

        self.conn.commit()
        return resolved_list

    def get_track_record(self) -> dict:
        """Get overall win/loss stats."""
        rows = self.conn.execute(
            """SELECT result, COUNT(*) FROM signals
               WHERE resolved = 1 GROUP BY result"""
        ).fetchall()

        stats = {"total": 0, "wins": 0, "losses": 0}
        for result, count in rows:
            stats["total"] += count
            if "win" in (result or ""):
                stats["wins"] += count
            else:
                stats["losses"] += count

        return stats

    def get_today_alerts_count(self) -> int:
        """Count alerts sent today."""
        today_start = int(
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        row = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE alerted = 1 AND created_at >= ?",
            (today_start,),
        ).fetchone()
        return row[0] if row else 0

    def get_last_alert_time(self, coin: str) -> Optional[float]:
        """Get timestamp of last alert for a coin."""
        row = self.conn.execute(
            """SELECT MAX(created_at) FROM signals
               WHERE coin = ? AND alerted = 1""",
            (coin,),
        ).fetchone()
        return row[0] if row and row[0] else None

    def get_daily_summary(self) -> dict:
        """Get today's stats for daily summary (enhanced)."""
        today_start = int(
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )

        total = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE created_at >= ?", (today_start,)
        ).fetchone()[0]

        longs = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE created_at >= ? AND direction = 'long'",
            (today_start,),
        ).fetchone()[0]

        wins = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE resolved = 1 AND resolved_at >= ? AND result LIKE '%win%'",
            (today_start,),
        ).fetchone()[0]

        losses = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE resolved = 1 AND resolved_at >= ? AND result LIKE '%loss%'",
            (today_start,),
        ).fetchone()[0]

        expired = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE resolved = 1 AND resolved_at >= ? AND result LIKE 'expired%'",
            (today_start,),
        ).fetchone()[0]

        avg_score = self.conn.execute(
            "SELECT AVG(score) FROM signals WHERE created_at >= ?", (today_start,)
        ).fetchone()[0] or 0

        avg_pnl = self.conn.execute(
            "SELECT AVG(pnl_pct) FROM signals WHERE resolved = 1 AND resolved_at >= ?",
            (today_start,),
        ).fetchone()[0] or 0

        # All-time stats
        all_time = self.get_track_record()
        all_wr = (all_time["wins"] / all_time["total"] * 100) if all_time["total"] > 0 else 0

        # Today win rate
        today_resolved = wins + losses
        today_wr = (wins / today_resolved * 100) if today_resolved > 0 else 0

        return {
            "total": total,
            "longs": longs,
            "shorts": total - longs,
            "wins": wins,
            "losses": losses,
            "expired": expired,
            "avg_score": avg_score,
            "avg_pnl_pct": avg_pnl * 100,
            "win_rate": today_wr,
            "all_time": {
                "total": all_time["total"],
                "wins": all_time["wins"],
                "losses": all_time["losses"],
                "win_rate": all_wr,
            },
        }

    def get_top_signals_today(self) -> list[dict]:
        """Get today's signals sorted by PnL for summary."""
        today_start = int(
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        rows = self.conn.execute(
            """SELECT coin, direction, pnl_pct, result, score
               FROM signals
               WHERE created_at >= ? AND resolved = 1
               ORDER BY pnl_pct DESC""",
            (today_start,),
        ).fetchall()

        return [
            {"coin": r[0], "direction": r[1], "pnl_pct": r[2] or 0,
             "result": r[3], "score": r[4]}
            for r in rows
        ]

    def get_active_signals(self, hl_data: HyperliquidData) -> list[dict]:
        """Get unresolved signals with current PnL."""
        rows = self.conn.execute(
            """SELECT coin, direction, entry_price, target_price, stop_price, score
               FROM signals WHERE resolved = 0
               ORDER BY created_at DESC"""
        ).fetchall()

        if not rows:
            return []

        try:
            mids = hl_data._post({"type": "allMids"})
        except:
            return []

        active = []
        for coin, direction, entry_px, target_px, stop_px, score in rows:
            current_px = float(mids.get(coin, 0))
            if current_px <= 0:
                continue
            if direction == "long":
                pnl = (current_px - entry_px) / entry_px
            else:
                pnl = (entry_px - current_px) / entry_px

            active.append({
                "coin": coin,
                "direction": direction,
                "entry_price": entry_px,
                "current_pnl_pct": pnl,
                "score": score,
            })

        return active


# =============================================================================
# MAIN BOT
# =============================================================================

class HyperPulseBot:
    """The HyperPulse alert bot."""

    def __init__(self, config: HyperPulseConfig):
        self.config = config
        self.conn = init_db(config.db_path)
        self.hl = HyperliquidData(config.hl_api_url)
        self.detector = SqueezeDetector()
        self.tracker = SignalTracker(self.conn, config)

        if config.dry_run:
            self.telegram = None
            log.info("DRY-RUN mode — no Telegram sending")
        else:
            self.telegram = TelegramSender(config.telegram_token, config.channel_id)

        # State
        self.last_data_refresh = 0.0
        self.candle_cache: dict[str, pd.DataFrame] = {}  # coin → DataFrame
        self.token_list: list[dict] = []

        # Scheduled messages state
        self._last_daily_summary_date: str = ""
        self._last_morning_briefing_date: str = ""

    def run(self):
        """Main loop."""
        print("=" * 60)
        print("  ⚡ HYPERPULSE — See the squeeze before it fires.")
        print(f"  Mode: {'DRY-RUN' if self.config.dry_run else 'LIVE'}")
        print(f"  Scan interval: {self.config.scan_interval_sec}s")
        print(f"  Min score: {self.config.min_squeeze_score}")
        print(f"  Min confidence: {self.config.min_direction_confidence}")
        if not self.config.dry_run:
            print(f"  Channel: {self.config.channel_id}")
        print("=" * 60)

        # Initial data load
        log.info("Loading initial data...")
        self._refresh_data()

        log.info(f"Bot started — scanning every {self.config.scan_interval_sec}s\n")

        try:
            while True:
                now = time.time()

                # Refresh data periodically
                if now - self.last_data_refresh >= self.config.data_refresh_sec:
                    self._refresh_data()

                # Scan for squeezes
                self._scan_and_alert()

                # Resolve old signals + send resolution alerts
                try:
                    resolved = self.tracker.resolve_signals(self.hl)
                    if resolved and self.config.send_resolution_alerts:
                        self._send_resolution_alerts(resolved)
                except Exception as e:
                    log.error(f"Error resolving signals: {e}")

                # Check scheduled messages (daily summary, morning briefing)
                self._check_scheduled_messages()

                # Sleep
                log.info(f"Next scan in {self.config.scan_interval_sec}s...")
                time.sleep(self.config.scan_interval_sec)

        except KeyboardInterrupt:
            log.info("\nShutdown requested.")
            summary = self.tracker.get_daily_summary()
            log.info(f"Summary: {summary['total']} signals, "
                     f"{summary['wins']}W/{summary['losses']}L, "
                     f"WR={summary['win_rate']:.0f}%")

    # =========================================================================
    # DATA REFRESH
    # =========================================================================

    def _refresh_data(self):
        """Refresh token list and candle data."""
        log.info("Refreshing Hyperliquid data...")

        try:
            # Get token list
            self.token_list = self.hl.get_all_tokens(
                min_vol=self.config.min_volume_24h,
                max_vol=self.config.max_volume_24h,
            )
            log.info(f"  {len(self.token_list)} tokens found")

            # Fetch candles for each token
            loaded = 0
            errors = 0
            for i, token in enumerate(self.token_list):
                coin = token["symbol"]
                try:
                    df = self.hl.fetch_candles(
                        coin, "1h", limit=self.config.warmup_candles
                    )
                    if len(df) >= 100:
                        self.candle_cache[coin] = df
                        loaded += 1
                    time.sleep(0.5)  # Rate limit
                except Exception as e:
                    errors += 1
                    if "429" in str(e):
                        log.warning("  Rate limited, waiting 5s...")
                        time.sleep(5)

                # Progress
                if (i + 1) % 50 == 0:
                    log.info(f"  Progress: {i+1}/{len(self.token_list)} "
                             f"({loaded} OK, {errors} errors)")

            self.last_data_refresh = time.time()
            log.info(f"  {loaded} tokens loaded ({errors} errors)\n")

        except Exception as e:
            log.error(f"Refresh error: {e}")
            traceback.print_exc()

    # =========================================================================
    # SQUEEZE SCANNING
    # =========================================================================

    def _scan_and_alert(self):
        """Scan all tokens and send alerts for qualifying signals."""
        log.info("Scanning squeezes...")

        # Get current funding rates
        try:
            funding_rates = self.hl.get_funding_rates()
        except:
            funding_rates = {}

        # Compute BTC trend for trend filter
        btc_trend = None
        if "BTC" in self.candle_cache and len(self.candle_cache["BTC"]) >= 50:
            try:
                btc_trend = compute_trend(self.candle_cache["BTC"])
                log.info(f"  BTC trend: {btc_trend.bias} (strength={btc_trend.strength:.0%})")
            except Exception as e:
                log.warning(f"  BTC trend computation failed: {e}")

        # Scan all cached tokens
        signals = []
        for coin, df in self.candle_cache.items():
            if len(df) < 100:
                continue
            try:
                signal = self.detector.analyze(
                    df, coin,
                    funding_rate=funding_rates.get(coin, 0.0),
                )
                if signal.phase != SqueezePhase.NO_SQUEEZE:
                    signals.append((signal, funding_rates.get(coin, 0.0)))
            except Exception:
                pass

        # Sort by score
        signals.sort(key=lambda x: x[0].score, reverse=True)

        # Filter actionable signals
        actionable = []
        blocked_count = 0
        for signal, funding in signals:
            if signal.phase.value not in self.config.required_phases:
                continue
            if signal.direction == BreakoutDirection.UNKNOWN:
                continue
            if signal.score < self.config.min_squeeze_score:
                continue
            if signal.expected_move_pct < self.config.min_expected_move_pct:
                continue

            # TTM squeeze bars filter
            if signal.ttm_squeeze_bars < self.config.min_ttm_squeeze_bars:
                continue

            # Apply funding adjustment
            conf_adj, aligned = compute_funding_adjustment(
                signal.direction, funding, self.config
            )
            adjusted_confidence = signal.direction_confidence + conf_adj

            if adjusted_confidence < self.config.min_direction_confidence:
                continue

            # TREND FILTER
            if btc_trend is not None:
                token_trend = compute_trend(self.candle_cache.get(signal.coin))
                blocked, reason = should_block_signal(
                    signal.direction.value, token_trend, btc_trend
                )
                if blocked:
                    blocked_count += 1
                    log.info(f"  Blocked {signal.coin} {signal.direction.value}: {reason}")
                    continue

            # STRUCTURE FILTER: require structural confluence
            struct_reason = ""
            try:
                structure = analyze_structure(self.candle_cache[signal.coin])
                take, struct_adj, struct_reason = should_take_signal(
                    signal.direction.value, structure
                )
                if not take:
                    blocked_count += 1
                    log.info(f"  Blocked {signal.coin} {signal.direction.value}: {struct_reason}")
                    continue
                adjusted_confidence += struct_adj
                # Build structure info string for alert
                parts = []
                if structure.structure_bias != "neutral":
                    parts.append(f"BOS {structure.structure_bias}")
                fresh_obs = [ob for ob in structure.order_blocks if ob.fresh]
                if fresh_obs:
                    ob = fresh_obs[0]
                    parts.append(f"fresh OB at ${ob.zone_bottom:.4f}")
                if structure.recent_sweep:
                    parts.append("liquidity sweep detected")
                struct_reason = " + ".join(parts) if parts else ""
            except Exception as e:
                log.warning(f"  Structure analysis failed for {signal.coin}: {e}")

            actionable.append((signal, funding, adjusted_confidence, aligned, struct_reason))

        # Log summary
        building = [s for s, _ in signals if s.phase == SqueezePhase.BUILDING]
        log.info(
            f"  {len(signals)} squeezes detected "
            f"({len(building)} building, {len(actionable)} actionable, "
            f"{blocked_count} blocked by trend/structure)"
        )

        # Process actionable signals
        for signal, funding, adj_conf, aligned, struct_info in actionable:
            self._process_signal(signal, funding, adj_conf, aligned, struct_info)

    def _process_signal(
        self,
        signal: SqueezeSignal,
        funding_rate: float,
        adj_confidence: float,
        funding_aligned: bool,
        structure_info: str = "",
    ):
        """Process a single actionable signal: record + alert."""
        coin = signal.coin

        # Cooldown check
        last_alert = self.tracker.get_last_alert_time(coin)
        if last_alert:
            elapsed_min = (time.time() - last_alert) / 60
            if elapsed_min < self.config.cooldown_per_token_min:
                return

        # Free tier daily limit check
        today_count = self.tracker.get_today_alerts_count()
        if today_count >= self.config.free_max_alerts_per_day:
            # In free mode, stop after N alerts
            # (premium would bypass this)
            return

        # Get current price for entry
        try:
            mids = self.hl._post({"type": "allMids"})
            entry_price = float(mids.get(coin, 0))
        except:
            return

        if entry_price <= 0:
            return

        # Calculate target and stop
        atr = signal.atr_value
        if atr <= 0:
            atr = entry_price * 0.02

        is_long = signal.direction == BreakoutDirection.LONG
        if is_long:
            stop_price = entry_price - 1.5 * atr
            target_price = entry_price + 3.0 * atr
        else:
            stop_price = entry_price + 1.5 * atr
            target_price = entry_price - 3.0 * atr

        # Record signal
        track = self.tracker.get_track_record()
        sid = self.tracker.record_signal(
            signal, entry_price, target_price, stop_price,
            funding_rate, funding_aligned, adj_confidence,
        )

        # Format message
        msg = format_signal_alert(
            signal, funding_rate, funding_aligned, adj_confidence,
            entry_price, target_price, stop_price, track,
            structure_info=structure_info,
        )

        # Send alert
        if self.config.dry_run:
            print("\n" + "=" * 50)
            print(msg.replace("*", "").replace("`", "").replace("_", ""))
            print("=" * 50 + "\n")
        elif self.telegram:
            success = self.telegram.send_message(msg)
            if success:
                log.info(f"Alert sent: {coin} {signal.direction.value} "
                         f"(score={signal.score:.2f}, conf={adj_confidence:.0%})")
            else:
                log.error(f"Failed to send alert for {coin}")

        self.tracker.mark_alerted(sid)

    # =========================================================================
    # RESOLUTION ALERTS
    # =========================================================================

    def _send_resolution_alerts(self, resolved: list[dict]):
        """Send alerts for each newly resolved signal."""
        for r in resolved:
            msg = format_resolution_alert(
                coin=r["coin"],
                direction=r["direction"],
                entry_price=r["entry_price"],
                exit_price=r["exit_price"],
                pnl_pct=r["pnl_pct"],
                result=r["result"],
                duration_hours=r["duration_hours"],
                score=r["score"],
            )

            result_emoji = "✅" if "win" in r["result"] else "❌"
            log.info(
                f"  {result_emoji} Resolved: {r['coin']} {r['direction']} "
                f"-> {r['result']} ({r['pnl_pct']*100:+.1f}%)"
            )

            if self.config.dry_run:
                print("\n" + "-" * 40)
                print(msg.replace("*", "").replace("`", "").replace("_", ""))
                print("-" * 40 + "\n")
            elif self.telegram:
                self.telegram.send_message(msg)

    # =========================================================================
    # SCHEDULED MESSAGES
    # =========================================================================

    def _check_scheduled_messages(self):
        """Check if daily summary or morning briefing should be sent."""
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")

        # Daily summary at configured hour (default 21h UTC = 23h Paris)
        if (
            now_utc.hour == self.config.daily_summary_hour_utc
            and self._last_daily_summary_date != today_str
        ):
            self._send_daily_summary()
            self._last_daily_summary_date = today_str

        # Morning briefing at configured hour (default 8h UTC = 10h Paris)
        if (
            now_utc.hour == self.config.morning_briefing_hour_utc
            and self._last_morning_briefing_date != today_str
        ):
            self._send_morning_briefing()
            self._last_morning_briefing_date = today_str

    def _send_daily_summary(self):
        """Compile and send the daily summary."""
        log.info("Sending daily summary...")

        try:
            stats = self.tracker.get_daily_summary()
            top_signals = self.tracker.get_top_signals_today()
            msg = format_daily_summary(stats, top_signals)

            if self.config.dry_run:
                print("\n" + "=" * 50)
                print("📊 DAILY SUMMARY")
                print(msg.replace("*", "").replace("`", "").replace("_", ""))
                print("=" * 50 + "\n")
            elif self.telegram:
                success = self.telegram.send_message(msg)
                if success:
                    log.info("  Daily summary sent")
                else:
                    log.error("  Failed to send daily summary")
        except Exception as e:
            log.error(f"Daily summary error: {e}")

    def _send_morning_briefing(self):
        """Compile and send morning briefing with watchlist."""
        log.info("Sending morning briefing...")

        try:
            # Get building squeezes from last scan
            funding_rates = {}
            try:
                funding_rates = self.hl.get_funding_rates()
            except:
                pass

            building_signals = []
            for coin, df in self.candle_cache.items():
                if len(df) < 100:
                    continue
                try:
                    signal = self.detector.analyze(
                        df, coin,
                        funding_rate=funding_rates.get(coin, 0.0),
                    )
                    if signal.phase == SqueezePhase.BUILDING and signal.score >= 0.4:
                        building_signals.append({
                            "coin": coin,
                            "score": signal.score,
                            "ttm_squeeze": signal.ttm_squeeze,
                            "ttm_bars": signal.ttm_squeeze_bars,
                            "bb_pct": signal.bb_width_percentile,
                        })
                except:
                    pass

            building_signals.sort(key=lambda s: s["score"], reverse=True)

            # Active signals
            active_signals = self.tracker.get_active_signals(self.hl)

            # Market stats
            avg_funding = (
                sum(funding_rates.values()) / len(funding_rates)
                if funding_rates else 0
            )
            market_stats = {
                "total_tokens": len(self.candle_cache),
                "avg_funding": avg_funding * 100,
            }

            msg = format_morning_briefing(building_signals, active_signals, market_stats)

            if self.config.dry_run:
                print("\n" + "=" * 50)
                print("☀️ MORNING BRIEFING")
                print(msg.replace("*", "").replace("`", "").replace("_", ""))
                print("=" * 50 + "\n")
            elif self.telegram:
                success = self.telegram.send_message(msg)
                if success:
                    log.info("  Morning briefing sent")
                else:
                    log.error("  Failed to send morning briefing")
        except Exception as e:
            log.error(f"Morning briefing error: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="⚡ HyperPulse — Squeeze alerts for Hyperliquid"
    )

    # Telegram
    parser.add_argument(
        "--telegram-token", type=str,
        default=os.environ.get("HYPERPULSE_TG_TOKEN", ""),
        help="Telegram bot token (or env HYPERPULSE_TG_TOKEN)",
    )
    parser.add_argument(
        "--channel-id", type=str,
        default=os.environ.get("HYPERPULSE_TG_CHANNEL", ""),
        help="Telegram channel ID (or env HYPERPULSE_TG_CHANNEL)",
    )

    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        help="Print alerts to terminal, no Telegram")

    # Scan params
    parser.add_argument("--scan-interval", type=int, default=300,
                        help="Seconds between scans (default: 300)")
    parser.add_argument("--min-score", type=float, default=0.80,
                        help="Min squeeze score (default: 0.80)")
    parser.add_argument("--min-confidence", type=float, default=0.85,
                        help="Min direction confidence (default: 0.85)")
    parser.add_argument("--min-volume", type=float, default=100_000,
                        help="Min 24h volume USD (default: 100K)")
    parser.add_argument("--max-volume", type=float, default=500_000_000,
                        help="Max 24h volume USD (default: 500M)")

    # Database
    parser.add_argument("--db", type=str, default="hyperpulse.db",
                        help="SQLite database path")

    # Maintenance
    parser.add_argument("--reset-signals", action="store_true",
                        help="Archive old signals and reset history")

    args = parser.parse_args()

    # Handle --reset-signals
    if args.reset_signals:
        _reset_signals(args.db)
        sys.exit(0)

    # Validate
    if not args.dry_run and not args.telegram_token:
        print("Telegram token required. Use --telegram-token or --dry-run")
        print("   Create a bot: https://t.me/BotFather")
        sys.exit(1)

    if not args.dry_run and not args.channel_id:
        print("Channel ID required. Use --channel-id or --dry-run")
        sys.exit(1)

    config = HyperPulseConfig(
        telegram_token=args.telegram_token,
        channel_id=args.channel_id,
        dry_run=args.dry_run,
        scan_interval_sec=args.scan_interval,
        min_squeeze_score=args.min_score,
        min_direction_confidence=args.min_confidence,
        min_volume_24h=args.min_volume,
        max_volume_24h=args.max_volume,
        db_path=args.db,
    )

    bot = HyperPulseBot(config)
    bot.run()


def _reset_signals(db_path: str):
    """Archive old signals and reset the signals table."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Create archive table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals_archive AS
        SELECT * FROM signals WHERE 0
    """)

    # Archive existing signals
    count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    if count > 0:
        conn.execute("INSERT INTO signals_archive SELECT * FROM signals")
        conn.execute("DELETE FROM signals")
        conn.commit()
        print(f"Signal history reset: {count} signals archived to signals_archive")
    else:
        print("No signals to reset")

    conn.close()


if __name__ == "__main__":
    main()
