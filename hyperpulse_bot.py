#!/usr/bin/env python3
"""
=============================================================================
 HYPERPULSE — See the squeeze before it fires.
=============================================================================
 Bot Telegram d'alertes squeeze pour Hyperliquid.

 Scanne toutes les paires Hyperliquid, détecte les squeezes,
 prédit la direction, ajuste avec les funding rates,
 et envoie des alertes Telegram formatées.

 Usage:
   # Premier lancement (collecte ~200 candles par token, ~10 min)
   python hyperpulse_bot.py --telegram-token YOUR_BOT_TOKEN --channel-id @your_channel

   # Mode dry-run (affiche dans le terminal, pas de Telegram)
   python hyperpulse_bot.py --dry-run

   # Config custom
   python hyperpulse_bot.py --telegram-token TOKEN --channel-id @chan \
       --scan-interval 300 --min-score 0.55 --min-confidence 0.60

 Prérequis:
   pip install requests pandas numpy python-telegram-bot --break-system-packages

 Structure:
   hyperpulse_bot.py    ← Ce fichier (le bot)
   squeeze_detector.py  ← Ton détecteur existant (inchangé)
   hyperpulse.db        ← Base SQLite (créée automatiquement)
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
    """Configuration du bot HyperPulse."""

    # --- Telegram ---
    telegram_token: str = ""
    channel_id: str = ""            # @channel_name ou -100xxxxx (chat ID)
    premium_channel_id: str = ""    # Channel premium (optionnel)

    # --- Mode ---
    dry_run: bool = False           # Pas de Telegram, print seulement

    # --- Scan ---
    scan_interval_sec: int = 300    # 5 minutes entre chaque scan
    data_refresh_sec: int = 900     # 15 min entre chaque refresh complet des candles
    warmup_candles: int = 200       # Nombre de candles pour warm-up des indicateurs

    # --- Filtres signaux ---
    min_squeeze_score: float = 0.55
    min_direction_confidence: float = 0.60
    min_expected_move_pct: float = 0.02
    required_phases: list = field(
        default_factory=lambda: ["ready", "firing"]
    )
    # Tokens: volume min/max pour filtrer
    min_volume_24h: float = 100_000
    max_volume_24h: float = 500_000_000

    # --- Funding rate adjustment ---
    funding_boost_max: float = 0.15     # ±15% confidence adjustment
    funding_strong_threshold: float = 0.0003  # 0.03%/h = signal fort

    # --- Signal tracking ---
    signal_ttl_hours: int = 24          # Résoudre les signaux après 24h
    cooldown_per_token_min: int = 120   # Min 2h entre deux alertes pour le même token

    # --- Free tier limits ---
    free_max_alerts_per_day: int = 5
    free_top_n_tokens: int = 15         # Top 15 par volume seulement

    # --- Database ---
    db_path: str = "hyperpulse.db"

    # --- Hyperliquid API ---
    hl_api_url: str = "https://api.hyperliquid.xyz/info"


# =============================================================================
# DATABASE
# =============================================================================

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialise la base SQLite pour HyperPulse."""
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
    - LONG + negative funding (shorts pay longs) = ALIGNED → boost
    - SHORT + positive funding (longs pay shorts) = ALIGNED → boost
    - Opposite = MISALIGNED → penalty
    """
    if direction == BreakoutDirection.UNKNOWN or funding_rate == 0:
        return 0.0, False

    is_long = direction == BreakoutDirection.LONG

    # Alignment: we want to trade in the direction that gets paid
    # Negative funding → shorts pay → LONG is paid → aligned with LONG
    # Positive funding → longs pay → SHORT is paid → aligned with SHORT
    if is_long:
        alignment = -funding_rate  # Positive if funding is negative (good for longs)
    else:
        alignment = funding_rate   # Positive if funding is positive (good for shorts)

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
) -> str:
    """Format a squeeze signal as a Telegram message."""

    is_long = signal.direction == BreakoutDirection.LONG
    dir_emoji = "🟢" if is_long else "🔴"
    dir_text = "LONG 📈" if is_long else "SHORT 📉"
    phase_text = signal.phase.value.upper()

    # Funding info
    funding_pct = funding_rate * 100
    if funding_aligned:
        funding_status = "✅ PAYÉ"
    elif abs(funding_rate) < 0.0001:
        funding_status = "➖ Neutre"
    else:
        funding_status = "⚠️ Contre"

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
        f"💵 Prix: `${entry_price:.4f}`\n"
        f"🎯 Target: `${target_price:.4f}` (+{target_dist:.1f}%)\n"
        f"🛑 Stop: `${stop_price:.4f}` (-{stop_dist:.1f}%)\n"
        f"\n"
        f"_Indicateurs:_\n"
        f"  ├ BB width: `P{signal.bb_width_percentile:.0f}` "
        f"{'✅' if signal.bb_width_percentile < 20 else '⚠️'}\n"
        f"  ├ TTM Squeeze: "
        f"{'✅ ' + str(signal.ttm_squeeze_bars) + ' bars' if signal.ttm_squeeze else '❌'}\n"
        f"  ├ ATR: `P{signal.atr_percentile:.0f}` "
        f"{'✅' if signal.atr_percentile < 25 else '⚠️'}\n"
        f"  ├ Volume ratio: `{signal.volume_ratio:.1f}x`\n"
        f"  └ Expected move: `{signal.expected_move_pct:.1%}`\n"
        f"\n"
        f"📊 Track record: {wr:.0f}% win rate ({total} signaux)"
    )

    return msg


def format_daily_summary(stats: dict) -> str:
    """Format daily summary message."""
    msg = (
        f"📊 *HyperPulse — Résumé du jour*\n"
        f"\n"
        f"Signaux émis: {stats.get('total', 0)}\n"
        f"  ├ Long: {stats.get('longs', 0)} | Short: {stats.get('shorts', 0)}\n"
        f"  ├ Résolus: {stats.get('resolved', 0)}\n"
        f"  ├ ✅ Wins: {stats.get('wins', 0)}\n"
        f"  ├ ❌ Losses: {stats.get('losses', 0)}\n"
        f"  └ ⏰ Expirés: {stats.get('expired', 0)}\n"
        f"\n"
        f"Win rate: {stats.get('win_rate', 0):.0f}%\n"
        f"Score moyen: {stats.get('avg_score', 0):.2f}\n"
        f"\n"
        f"_hyper-pulse.xyz — See the squeeze before it fires._"
    )
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

    def resolve_signals(self, hl_data: HyperliquidData):
        """Auto-resolve open signals: check if target/stop hit or TTL expired."""
        open_signals = self.conn.execute(
            """SELECT id, coin, direction, entry_price, target_price,
                      stop_price, timestamp
               FROM signals WHERE resolved = 0"""
        ).fetchall()

        now = time.time()

        for row in open_signals:
            sid, coin, direction, entry_px, target_px, stop_px, ts = row
            age_hours = (now - ts) / 3600

            # Get current price
            try:
                mids = hl_data._post({"type": "allMids"})
                current_price = float(mids.get(coin, 0))
            except:
                continue

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
                # Determine win/loss based on direction of price movement
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

        self.conn.commit()

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
        """Get today's stats for daily summary."""
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

        resolved = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE resolved = 1 AND resolved_at >= ?",
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

        track = self.get_track_record()
        wr = (track["wins"] / track["total"] * 100) if track["total"] > 0 else 0

        return {
            "total": total,
            "longs": longs,
            "shorts": total - longs,
            "resolved": resolved,
            "wins": wins,
            "losses": losses,
            "expired": expired,
            "avg_score": avg_score,
            "win_rate": wr,
        }


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
            log.info("🔵 Mode DRY-RUN — pas d'envoi Telegram")
        else:
            self.telegram = TelegramSender(config.telegram_token, config.channel_id)

        # State
        self.last_data_refresh = 0.0
        self.candle_cache: dict[str, pd.DataFrame] = {}  # coin → DataFrame
        self.token_list: list[dict] = []

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
        log.info("📊 Chargement initial des données...")
        self._refresh_data()

        log.info(f"🚀 Bot démarré — scan toutes les {self.config.scan_interval_sec}s\n")

        try:
            while True:
                now = time.time()

                # Refresh data periodically
                if now - self.last_data_refresh >= self.config.data_refresh_sec:
                    self._refresh_data()

                # Scan for squeezes
                self._scan_and_alert()

                # Resolve old signals
                try:
                    self.tracker.resolve_signals(self.hl)
                except Exception as e:
                    log.error(f"Error resolving signals: {e}")

                # Sleep
                log.info(f"💤 Prochain scan dans {self.config.scan_interval_sec}s...")
                time.sleep(self.config.scan_interval_sec)

        except KeyboardInterrupt:
            log.info("\n⛔ Arrêt demandé.")
            summary = self.tracker.get_daily_summary()
            log.info(f"📊 Résumé: {summary['total']} signaux, "
                     f"{summary['wins']}W/{summary['losses']}L, "
                     f"WR={summary['win_rate']:.0f}%")

    # =========================================================================
    # DATA REFRESH
    # =========================================================================

    def _refresh_data(self):
        """Refresh token list and candle data."""
        log.info("📊 Refresh des données Hyperliquid...")

        try:
            # Get token list
            self.token_list = self.hl.get_all_tokens(
                min_vol=self.config.min_volume_24h,
                max_vol=self.config.max_volume_24h,
            )
            log.info(f"  {len(self.token_list)} tokens trouvés")

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
                             f"({loaded} OK, {errors} erreurs)")

            self.last_data_refresh = time.time()
            log.info(f"  ✅ {loaded} tokens chargés ({errors} erreurs)\n")

        except Exception as e:
            log.error(f"Erreur refresh: {e}")
            traceback.print_exc()

    # =========================================================================
    # SQUEEZE SCANNING
    # =========================================================================

    def _scan_and_alert(self):
        """Scan all tokens and send alerts for qualifying signals."""
        log.info("🔍 Scanning squeezes...")

        # Get current funding rates
        try:
            funding_rates = self.hl.get_funding_rates()
        except:
            funding_rates = {}

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
        for signal, funding in signals:
            if signal.phase.value not in self.config.required_phases:
                continue
            if signal.direction == BreakoutDirection.UNKNOWN:
                continue
            if signal.score < self.config.min_squeeze_score:
                continue
            if signal.expected_move_pct < self.config.min_expected_move_pct:
                continue

            # Apply funding adjustment
            conf_adj, aligned = compute_funding_adjustment(
                signal.direction, funding, self.config
            )
            adjusted_confidence = signal.direction_confidence + conf_adj

            if adjusted_confidence < self.config.min_direction_confidence:
                continue

            actionable.append((signal, funding, adjusted_confidence, aligned))

        # Log summary
        building = [s for s, _ in signals if s.phase == SqueezePhase.BUILDING]
        log.info(
            f"  {len(signals)} squeezes détectés "
            f"({len(building)} building, {len(actionable)} actionables)"
        )

        # Process actionable signals
        for signal, funding, adj_conf, aligned in actionable:
            self._process_signal(signal, funding, adj_conf, aligned)

    def _process_signal(
        self,
        signal: SqueezeSignal,
        funding_rate: float,
        adj_confidence: float,
        funding_aligned: bool,
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
        )

        # Send alert
        if self.config.dry_run:
            print("\n" + "=" * 50)
            print(msg.replace("*", "").replace("`", "").replace("_", ""))
            print("=" * 50 + "\n")
        elif self.telegram:
            success = self.telegram.send_message(msg)
            if success:
                log.info(f"📤 Alerte envoyée: {coin} {signal.direction.value} "
                         f"(score={signal.score:.2f}, conf={adj_confidence:.0%})")
            else:
                log.error(f"❌ Échec envoi alerte {coin}")

        self.tracker.mark_alerted(sid)


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
    parser.add_argument("--min-score", type=float, default=0.55,
                        help="Min squeeze score (default: 0.55)")
    parser.add_argument("--min-confidence", type=float, default=0.60,
                        help="Min direction confidence (default: 0.60)")
    parser.add_argument("--min-volume", type=float, default=100_000,
                        help="Min 24h volume USD (default: 100K)")
    parser.add_argument("--max-volume", type=float, default=500_000_000,
                        help="Max 24h volume USD (default: 500M)")

    # Database
    parser.add_argument("--db", type=str, default="hyperpulse.db",
                        help="SQLite database path")

    args = parser.parse_args()

    # Validate
    if not args.dry_run and not args.telegram_token:
        print("❌ Telegram token required. Use --telegram-token or --dry-run")
        print("   Créer un bot: https://t.me/BotFather")
        sys.exit(1)

    if not args.dry_run and not args.channel_id:
        print("❌ Channel ID required. Use --channel-id or --dry-run")
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


if __name__ == "__main__":
    main()
