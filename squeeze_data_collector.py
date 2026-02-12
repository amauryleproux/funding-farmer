"""
=============================================================================
SQUEEZE DATA COLLECTOR — Hyperliquid + Solana (GeckoTerminal)
=============================================================================
Collecte OHLCV + funding rates pour détecter les squeezes sur :
  1. Hyperliquid : tous les perps (focus small/mid-cap)
  2. Solana : tokens DEX via GeckoTerminal (gratuit, pas de clé API)

Stockage SQLite local. Compatible avec le SqueezeDetector.

Usage:
  # Collecte initiale complète (peut prendre 10-15 min)
  python squeeze_data_collector.py --collect-all

  # Collecte incrémentale (à lancer en cron toutes les heures)
  python squeeze_data_collector.py --update

  # Scanner les squeezes sur les données collectées
  python squeeze_data_collector.py --scan

  # Tout d'un coup : update + scan
  python squeeze_data_collector.py --update --scan

  # Mode continu (update + scan en boucle)
  python squeeze_data_collector.py --live --interval 300
=============================================================================
"""

import argparse
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("squeeze_collector")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration du collecteur."""

    # --- Database ---
    db_path: str = "squeeze_data.db"

    # --- Hyperliquid ---
    hl_base_url: str = "https://api.hyperliquid.xyz/info"
    hl_intervals: list[str] = field(default_factory=lambda: ["1h"])
    hl_candle_limit: int = 5000  # Max par requête
    hl_min_volume_24h: float = 100_000  # $100K min volume
    hl_max_volume_24h: float = 50_000_000  # $50M max (éviter BTC/ETH)
    hl_request_delay: float = 1.0  # Délai entre requêtes (rate limit safe)

    # --- Solana / GeckoTerminal ---
    gt_base_url: str = "https://api.geckoterminal.com/api/v2"
    gt_network: str = "solana"
    gt_intervals: list[str] = field(default_factory=lambda: ["hour"])
    gt_trending_limit: int = 30  # Top N trending pools
    gt_min_volume_24h: float = 50_000  # $50K min
    gt_request_delay: float = 1.5  # GeckoTerminal rate limit (30 req/min)

    # --- Tokens Solana prédéfinis (les plus intéressants pour le squeeze) ---
    # Format: {name: pool_address} (adresse de la pool Raydium/Jupiter)
    solana_watchlist: dict[str, str] = field(default_factory=lambda: {
        # On remplira dynamiquement via trending, mais voici quelques fixes
    })

    # --- Squeeze Detection ---
    min_candles_for_analysis: int = 100


# =============================================================================
# DATABASE
# =============================================================================

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialise la base SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candles (
            source      TEXT NOT NULL,       -- 'hyperliquid' ou 'solana'
            symbol      TEXT NOT NULL,       -- coin name ou pool address
            display_name TEXT,               -- nom lisible
            interval    TEXT NOT NULL,       -- '1h', '4h', etc.
            timestamp   INTEGER NOT NULL,    -- epoch ms
            open        REAL NOT NULL,
            high        REAL NOT NULL,
            low         REAL NOT NULL,
            close       REAL NOT NULL,
            volume      REAL NOT NULL,
            num_trades  INTEGER DEFAULT 0,
            PRIMARY KEY (source, symbol, interval, timestamp)
        );

        CREATE TABLE IF NOT EXISTS token_meta (
            source      TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            display_name TEXT,
            volume_24h  REAL DEFAULT 0,
            market_cap  REAL DEFAULT 0,
            funding_rate REAL DEFAULT 0,    -- Hyperliquid only
            open_interest REAL DEFAULT 0,   -- Hyperliquid only
            pool_address TEXT,              -- Solana only
            dex         TEXT,               -- Solana only (Jupiter, Raydium...)
            last_updated INTEGER DEFAULT 0,
            PRIMARY KEY (source, symbol)
        );

        CREATE TABLE IF NOT EXISTS squeeze_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            display_name TEXT,
            timestamp   INTEGER NOT NULL,
            phase       TEXT NOT NULL,
            score       REAL NOT NULL,
            direction   TEXT,
            confidence  REAL DEFAULT 0,
            bb_width_pct REAL,
            ttm_squeeze INTEGER DEFAULT 0,
            ttm_bars    INTEGER DEFAULT 0,
            atr_pct     REAL,
            volume_ratio REAL,
            expected_move REAL,
            funding_rate REAL DEFAULT 0,
            created_at  INTEGER DEFAULT (strftime('%s','now'))
        );

        CREATE INDEX IF NOT EXISTS idx_candles_lookup
            ON candles(source, symbol, interval, timestamp);

        CREATE INDEX IF NOT EXISTS idx_signals_recent
            ON squeeze_signals(timestamp DESC, score DESC);
    """)
    conn.commit()
    return conn


def save_candles(
    conn: sqlite3.Connection,
    source: str,
    symbol: str,
    display_name: str,
    interval: str,
    candles: list[dict],
):
    """Insère des candles en batch (INSERT OR REPLACE)."""
    if not candles:
        return

    conn.executemany(
        """INSERT OR REPLACE INTO candles
           (source, symbol, display_name, interval, timestamp,
            open, high, low, close, volume, num_trades)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                source, symbol, display_name, interval,
                c["t"], c["o"], c["h"], c["l"], c["c"], c["v"],
                c.get("n", 0),
            )
            for c in candles
        ],
    )
    conn.commit()


def save_token_meta(conn: sqlite3.Connection, meta: dict):
    """Upsert les métadonnées d'un token."""
    conn.execute(
        """INSERT OR REPLACE INTO token_meta
           (source, symbol, display_name, volume_24h, market_cap,
            funding_rate, open_interest, pool_address, dex, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            meta["source"], meta["symbol"], meta.get("display_name", ""),
            meta.get("volume_24h", 0), meta.get("market_cap", 0),
            meta.get("funding_rate", 0), meta.get("open_interest", 0),
            meta.get("pool_address", ""), meta.get("dex", ""),
            int(time.time() * 1000),
        ),
    )
    conn.commit()


def load_candles_df(
    conn: sqlite3.Connection,
    source: str,
    symbol: str,
    interval: str = "1h",
    limit: int = 5000,
) -> pd.DataFrame:
    """Charge les candles en DataFrame pour l'analyse."""
    df = pd.read_sql_query(
        """SELECT timestamp as t, open, high, low, close, volume
           FROM candles
           WHERE source = ? AND symbol = ? AND interval = ?
           ORDER BY timestamp DESC
           LIMIT ?""",
        conn,
        params=(source, symbol, interval, limit),
    )
    if df.empty:
        return df

    df = df.sort_values("t").reset_index(drop=True)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df


def get_latest_timestamp(
    conn: sqlite3.Connection,
    source: str,
    symbol: str,
    interval: str = "1h",
) -> Optional[int]:
    """Retourne le timestamp le plus récent pour un symbole."""
    row = conn.execute(
        """SELECT MAX(timestamp) FROM candles
           WHERE source = ? AND symbol = ? AND interval = ?""",
        (source, symbol, interval),
    ).fetchone()
    return row[0] if row and row[0] else None


# =============================================================================
# HYPERLIQUID DATA FETCHER
# =============================================================================

class HyperliquidFetcher:
    """Collecteur de données Hyperliquid."""

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def _post(self, payload: dict) -> dict | list:
        """POST vers l'API info."""
        resp = self.session.post(self.config.hl_base_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_all_tokens(self) -> list[dict]:
        """
        Récupère tous les tokens perp avec leurs métriques.
        Retourne une liste triée par volume 24h.
        """
        # Meta (liste des assets)
        meta = self._post({"type": "meta"})
        universe = meta.get("universe", [])

        # Asset contexts (volumes, OI, funding)
        ctx_resp = self._post({"type": "metaAndAssetCtxs"})
        contexts = ctx_resp[1] if isinstance(ctx_resp, list) and len(ctx_resp) > 1 else []

        tokens = []
        for i, asset in enumerate(universe):
            name = asset.get("name", f"UNKNOWN_{i}")
            if i < len(contexts):
                ctx = contexts[i]
                volume_24h = float(ctx.get("dayNtlVlm", 0))
                funding = float(ctx.get("funding", 0))
                oi = float(ctx.get("openInterest", 0))
                mark_px = float(ctx.get("markPx", 0))
            else:
                volume_24h = 0
                funding = 0
                oi = 0
                mark_px = 0

            tokens.append({
                "source": "hyperliquid",
                "symbol": name,
                "display_name": name,
                "volume_24h": volume_24h,
                "funding_rate": funding,
                "open_interest": oi * mark_px if mark_px else 0,
                "mark_price": mark_px,
                "index": i,
            })

        # Filtrer et trier par volume
        tokens = [
            t for t in tokens
            if self.config.hl_min_volume_24h <= t["volume_24h"] <= self.config.hl_max_volume_24h
        ]
        tokens.sort(key=lambda t: t["volume_24h"], reverse=True)

        log.info(
            f"[HL] {len(tokens)} tokens dans la range "
            f"${self.config.hl_min_volume_24h/1e3:.0f}K - ${self.config.hl_max_volume_24h/1e6:.0f}M volume"
        )
        return tokens

    def fetch_candles(
        self,
        coin: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[dict]:
        """
        Récupère les candles pour un token.
        Max 5000 par requête.
        """
        now_ms = int(time.time() * 1000)

        if end_time is None:
            end_time = now_ms

        if start_time is None:
            # Par défaut : remonter aussi loin que possible
            # 1h candles : 5000 * 3600 * 1000 = ~208 jours
            interval_ms = self._interval_to_ms(interval)
            start_time = end_time - (self.config.hl_candle_limit * interval_ms)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
            },
        }

        try:
            raw = self._post(payload)
        except Exception as e:
            log.error(f"[HL] Erreur candles {coin}: {e}")
            return []

        if not isinstance(raw, list):
            return []

        candles = []
        for c in raw:
            candles.append({
                "t": int(c.get("t", 0)),
                "o": float(c.get("o", 0)),
                "h": float(c.get("h", 0)),
                "l": float(c.get("l", 0)),
                "c": float(c.get("c", 0)),
                "v": float(c.get("v", 0)),
                "n": int(c.get("n", 0)),
            })

        return candles

    def fetch_funding_rates(self) -> dict[str, float]:
        """Récupère les funding rates actuels pour tous les tokens."""
        try:
            ctx_resp = self._post({"type": "metaAndAssetCtxs"})
            meta = ctx_resp[0] if isinstance(ctx_resp, list) else {}
            contexts = ctx_resp[1] if isinstance(ctx_resp, list) and len(ctx_resp) > 1 else []
            universe = meta.get("universe", [])

            rates = {}
            for i, asset in enumerate(universe):
                name = asset.get("name", "")
                if i < len(contexts):
                    rates[name] = float(contexts[i].get("funding", 0))
            return rates
        except Exception as e:
            log.error(f"[HL] Erreur funding rates: {e}")
            return {}

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        mapping = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "8h": 28_800_000, "12h": 43_200_000,
            "1d": 86_400_000,
        }
        return mapping.get(interval, 3_600_000)


# =============================================================================
# SOLANA / GECKOTERMINAL DATA FETCHER
# =============================================================================

class SolanaFetcher:
    """
    Collecteur de données Solana via GeckoTerminal (gratuit, pas de clé API).

    GeckoTerminal agrège les données de tous les DEX Solana :
    Jupiter, Raydium, Orca, Meteora, etc.
    """

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json;version=20230302",
        })

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """GET vers GeckoTerminal API."""
        url = f"{self.config.gt_base_url}{endpoint}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_trending_pools(self, limit: int = None) -> list[dict]:
        """
        Récupère les pools trending sur Solana.
        Ce sont souvent les tokens avec le plus d'activité = potentiel squeeze.
        """
        limit = limit or self.config.gt_trending_limit

        try:
            data = self._get(
                f"/networks/{self.config.gt_network}/trending_pools",
                params={"page": 1},
            )
        except Exception as e:
            log.error(f"[SOL] Erreur trending pools: {e}")
            return []

        pools = []
        for item in data.get("data", [])[:limit]:
            attrs = item.get("attributes", {})
            name = attrs.get("name", "Unknown")
            address = attrs.get("address", "")
            volume_24h = float(attrs.get("volume_usd", {}).get("h24", 0) or 0)
            base_token = attrs.get("base_token_price_usd")

            if volume_24h < self.config.gt_min_volume_24h:
                continue

            # Extraire le nom du token de base
            token_name = name.split("/")[0].strip() if "/" in name else name

            pools.append({
                "source": "solana",
                "symbol": address,
                "display_name": token_name,
                "pool_name": name,
                "pool_address": address,
                "volume_24h": volume_24h,
                "base_price_usd": float(base_token) if base_token else 0,
                "dex": attrs.get("dex_id", "unknown"),
            })

        log.info(f"[SOL] {len(pools)} trending pools (>{self.config.gt_min_volume_24h/1e3:.0f}K vol)")
        return pools

    def get_top_pools_for_token(self, token_address: str) -> list[dict]:
        """Récupère les meilleures pools pour un token donné."""
        try:
            data = self._get(
                f"/networks/{self.config.gt_network}/tokens/{token_address}/pools",
                params={"sort": "h24_volume_usd_liquidity_desc", "page": 1},
            )
            pools = []
            for item in data.get("data", [])[:5]:
                attrs = item.get("attributes", {})
                pools.append({
                    "address": attrs.get("address", ""),
                    "name": attrs.get("name", ""),
                    "volume_24h": float(attrs.get("volume_usd", {}).get("h24", 0) or 0),
                    "dex": attrs.get("dex_id", ""),
                })
            return pools
        except Exception as e:
            log.error(f"[SOL] Erreur pools pour {token_address}: {e}")
            return []

    def fetch_candles(
        self,
        pool_address: str,
        timeframe: str = "hour",
        aggregate: int = 1,
        limit: int = 1000,
        before_timestamp: Optional[int] = None,
    ) -> list[dict]:
        """
        Récupère les candles OHLCV pour une pool Solana.

        Timeframes GeckoTerminal:
          - "minute" (1, 5, 15)
          - "hour" (1, 4, 12)
          - "day" (1)

        Max 1000 candles par requête.
        Pour plus, on pagine avec before_timestamp.
        """
        params = {
            "aggregate": aggregate,
            "limit": min(limit, 1000),
        }
        if before_timestamp:
            params["before_timestamp"] = before_timestamp

        endpoint = (
            f"/networks/{self.config.gt_network}/pools/{pool_address}"
            f"/ohlcv/{timeframe}"
        )

        try:
            data = self._get(endpoint, params=params)
        except Exception as e:
            log.error(f"[SOL] Erreur candles {pool_address}: {e}")
            return []

        ohlcv_list = (
            data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        )

        candles = []
        for c in ohlcv_list:
            if len(c) >= 6:
                candles.append({
                    "t": int(c[0]) * 1000,  # Convert to ms
                    "o": float(c[1]),
                    "h": float(c[2]),
                    "l": float(c[3]),
                    "c": float(c[4]),
                    "v": float(c[5]),
                    "n": 0,
                })

        return candles

    def fetch_full_history(
        self,
        pool_address: str,
        timeframe: str = "hour",
        max_candles: int = 5000,
    ) -> list[dict]:
        """
        Récupère l'historique complet en paginant.
        GeckoTerminal retourne max 1000 candles par requête.
        """
        all_candles = []
        before_ts = None

        while len(all_candles) < max_candles:
            batch = self.fetch_candles(
                pool_address,
                timeframe=timeframe,
                limit=1000,
                before_timestamp=before_ts,
            )

            if not batch:
                break

            all_candles.extend(batch)

            # Prochain cursor = timestamp le plus ancien
            oldest = min(c["t"] for c in batch) // 1000
            if before_ts and oldest >= before_ts:
                break  # On avance plus
            before_ts = oldest

            log.debug(f"  Fetched {len(batch)} candles, total {len(all_candles)}")
            time.sleep(self.config.gt_request_delay)

        # Déduplier et trier
        seen = set()
        unique = []
        for c in all_candles:
            if c["t"] not in seen:
                seen.add(c["t"])
                unique.append(c)
        unique.sort(key=lambda x: x["t"])

        return unique[:max_candles]


# =============================================================================
# MAIN COLLECTOR ORCHESTRATOR
# =============================================================================

class SqueezeDataCollector:
    """Orchestrateur principal : collecte + stockage + scan."""

    def __init__(self, config: CollectorConfig = None):
        self.config = config or CollectorConfig()
        self.conn = init_db(self.config.db_path)
        self.hl = HyperliquidFetcher(self.config)
        self.sol = SolanaFetcher(self.config)

    def collect_hyperliquid(self, full: bool = True):
        """
        Collecte les données Hyperliquid.
        full=True : historique complet (5000 candles)
        full=False : incrémental depuis le dernier timestamp
        """
        log.info("=" * 60)
        log.info("[HL] Début de la collecte Hyperliquid")
        log.info("=" * 60)

        # 1. Lister les tokens
        tokens = self.hl.get_all_tokens()
        log.info(f"[HL] {len(tokens)} tokens à collecter")

        # 2. Sauvegarder les métadonnées
        for t in tokens:
            save_token_meta(self.conn, t)

        # 3. Collecter les candles
        for i, token in enumerate(tokens):
            coin = token["symbol"]
            log.info(
                f"[HL] [{i+1}/{len(tokens)}] {coin} "
                f"(vol=${token['volume_24h']/1e6:.1f}M, "
                f"funding={token['funding_rate']*100:.4f}%)"
            )

            for interval in self.config.hl_intervals:
                if full:
                    start_time = None  # Remonter au max
                else:
                    latest = get_latest_timestamp(
                        self.conn, "hyperliquid", coin, interval
                    )
                    start_time = latest + 1 if latest else None

                candles = self.hl.fetch_candles(
                    coin, interval=interval, start_time=start_time
                )

                if candles:
                    save_candles(
                        self.conn, "hyperliquid", coin, coin, interval, candles
                    )
                    log.info(f"  → {len(candles)} candles {interval} sauvegardées")
                else:
                    log.warning(f"  → Aucune candle pour {coin}")

                time.sleep(self.config.hl_request_delay)

        # 4. Funding rates
        log.info("[HL] Récupération des funding rates...")
        rates = self.hl.fetch_funding_rates()
        for coin, rate in rates.items():
            self.conn.execute(
                """UPDATE token_meta SET funding_rate = ?, last_updated = ?
                   WHERE source = 'hyperliquid' AND symbol = ?""",
                (rate, int(time.time() * 1000), coin),
            )
        self.conn.commit()
        log.info(f"[HL] {len(rates)} funding rates mises à jour")

    def collect_solana(self, full: bool = True):
        """
        Collecte les données Solana via GeckoTerminal.
        """
        log.info("=" * 60)
        log.info("[SOL] Début de la collecte Solana")
        log.info("=" * 60)

        # 1. Récupérer les trending pools
        pools = self.sol.get_trending_pools()
        log.info(f"[SOL] {len(pools)} pools à collecter")

        # 2. Sauvegarder les métadonnées
        for p in pools:
            save_token_meta(self.conn, p)

        # 3. Collecter les candles
        for i, pool in enumerate(pools):
            addr = pool["pool_address"]
            name = pool["display_name"]
            log.info(
                f"[SOL] [{i+1}/{len(pools)}] {name} ({pool['dex']}) "
                f"vol=${pool['volume_24h']/1e3:.0f}K"
            )

            if full:
                candles = self.sol.fetch_full_history(addr, timeframe="hour")
            else:
                # Incrémental : juste les dernières 100 candles
                candles = self.sol.fetch_candles(addr, timeframe="hour", limit=100)

            if candles:
                save_candles(
                    self.conn, "solana", addr, name, "1h", candles
                )
                log.info(f"  → {len(candles)} candles sauvegardées")
            else:
                log.warning(f"  → Aucune candle pour {name}")

            time.sleep(self.config.gt_request_delay)

    def scan_squeezes(self) -> list[dict]:
        """
        Lance le SqueezeDetector sur toutes les données collectées.
        Retourne les signaux triés par score.
        """
        # Import du détecteur
        try:
            from squeeze_detector import (
                SqueezeDetector, SqueezeConfig, SqueezePhase
            )
        except ImportError:
            log.error(
                "squeeze_detector.py non trouvé ! "
                "Place-le dans le même dossier."
            )
            return []

        log.info("=" * 60)
        log.info("SCAN DES SQUEEZES")
        log.info("=" * 60)

        detector = SqueezeDetector()
        signals = []

        # Récupérer tous les tokens avec des candles
        rows = self.conn.execute(
            """SELECT DISTINCT source, symbol, display_name
               FROM candles
               GROUP BY source, symbol
               HAVING COUNT(*) >= ?""",
            (self.config.min_candles_for_analysis,),
        ).fetchall()

        log.info(f"Analyse de {len(rows)} tokens...")

        # Funding rates
        funding_rates = {}
        for row in self.conn.execute(
            "SELECT symbol, funding_rate FROM token_meta WHERE source = 'hyperliquid'"
        ).fetchall():
            funding_rates[row[0]] = row[1]

        for source, symbol, display_name in rows:
            df = load_candles_df(self.conn, source, symbol, "1h")
            if df.empty or len(df) < self.config.min_candles_for_analysis:
                continue

            try:
                funding = funding_rates.get(symbol, 0.0)
                signal = detector.analyze(df, display_name or symbol, funding_rate=funding)

                # Sauvegarder si intéressant
                if signal.phase not in (SqueezePhase.NO_SQUEEZE,):
                    self.conn.execute(
                        """INSERT INTO squeeze_signals
                           (source, symbol, display_name, timestamp, phase,
                            score, direction, confidence, bb_width_pct,
                            ttm_squeeze, ttm_bars, atr_pct, volume_ratio,
                            expected_move, funding_rate)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            source, symbol, display_name,
                            int(time.time() * 1000),
                            signal.phase.value, signal.score,
                            signal.direction.value, signal.direction_confidence,
                            signal.bb_width_percentile,
                            1 if signal.ttm_squeeze else 0,
                            signal.ttm_squeeze_bars,
                            signal.atr_percentile, signal.volume_ratio,
                            signal.expected_move_pct, signal.current_funding,
                        ),
                    )

                    signals.append({
                        "source": source,
                        "symbol": symbol,
                        "display_name": display_name,
                        "phase": signal.phase.value,
                        "score": signal.score,
                        "direction": signal.direction.value,
                        "confidence": signal.direction_confidence,
                        "expected_move": signal.expected_move_pct,
                        "funding": funding,
                        "ttm_squeeze": signal.ttm_squeeze,
                        "ttm_bars": signal.ttm_squeeze_bars,
                        "volume_ratio": signal.volume_ratio,
                    })

            except Exception as e:
                log.debug(f"  Erreur analyse {display_name}: {e}")

        self.conn.commit()

        # Trier par score
        signals.sort(key=lambda s: s["score"], reverse=True)

        # Afficher le dashboard
        self._print_dashboard(signals)

        return signals

    def _print_dashboard(self, signals: list[dict]):
        """Affiche le dashboard des squeezes détectés."""
        print("\n" + "=" * 80)
        print("🔍 SQUEEZE DASHBOARD")
        print("=" * 80)

        if not signals:
            print("  Aucun squeeze détecté actuellement.")
            return

        # Phase 1 candidates (READY / FIRING)
        phase1 = [
            s for s in signals
            if s["phase"] in ("ready", "firing")
            and s["direction"] != "unknown"
            and s["confidence"] >= 0.6
        ]

        # Phase 2 candidates (EXPANSION + high funding)
        phase2 = [
            s for s in signals
            if s["phase"] in ("expansion", "firing")
            and abs(s["funding"]) > 0.0003
        ]

        # Building (à surveiller)
        building = [s for s in signals if s["phase"] == "building" and s["score"] >= 0.4]

        print(f"\n🚀 PHASE 1 — TRADES DIRECTIONNELS ({len(phase1)} candidats)")
        print("-" * 80)
        if phase1:
            print(f"  {'Token':<15} {'Source':<10} {'Phase':<10} {'Score':>6} "
                  f"{'Dir':>6} {'Conf':>6} {'Move%':>7} {'TTM':>4} {'VolR':>5}")
            print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*4} {'-'*5}")
            for s in phase1[:15]:
                ttm = "✅" if s["ttm_squeeze"] else "❌"
                print(
                    f"  {s['display_name']:<15} {s['source']:<10} {s['phase']:<10} "
                    f"{s['score']:>6.2f} {s['direction']:>6} {s['confidence']:>5.0%} "
                    f"{s['expected_move']:>6.1%} {ttm:>4} {s['volume_ratio']:>5.1f}"
                )
        else:
            print("  Aucun candidat Phase 1 pour l'instant.")

        print(f"\n💰 PHASE 2 — FUNDING FARMING ({len(phase2)} candidats)")
        print("-" * 80)
        if phase2:
            print(f"  {'Token':<15} {'Source':<10} {'Phase':<10} {'Funding/h':>10} {'Score':>6}")
            print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
            for s in phase2[:10]:
                print(
                    f"  {s['display_name']:<15} {s['source']:<10} {s['phase']:<10} "
                    f"{s['funding']:>9.4%} {s['score']:>6.2f}"
                )
        else:
            print("  Aucun candidat Phase 2 pour l'instant.")

        print(f"\n⏳ EN CONSTRUCTION ({len(building)} tokens à surveiller)")
        print("-" * 80)
        if building:
            for s in building[:20]:
                ttm = "TTM" if s["ttm_squeeze"] else "   "
                print(
                    f"  {s['display_name']:<15} {s['source']:<10} "
                    f"score={s['score']:.2f} {ttm} bars={s['ttm_bars']}"
                )

        print("\n" + "=" * 80)

    def get_db_stats(self):
        """Affiche les stats de la base."""
        print("\n📊 DATABASE STATS")
        print("-" * 40)

        # Candles par source
        for source in ["hyperliquid", "solana"]:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM candles WHERE source = ?", (source,)
            ).fetchone()[0]
            tokens = self.conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM candles WHERE source = ?", (source,)
            ).fetchone()[0]
            print(f"  {source}: {count:,} candles, {tokens} tokens")

        # Tokens meta
        meta_count = self.conn.execute("SELECT COUNT(*) FROM token_meta").fetchone()[0]
        print(f"  Token metadata: {meta_count}")

        # Signaux récents
        recent = self.conn.execute(
            """SELECT COUNT(*) FROM squeeze_signals
               WHERE created_at > strftime('%s','now') - 3600"""
        ).fetchone()[0]
        print(f"  Signaux dernière heure: {recent}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Squeeze Data Collector — Hyperliquid + Solana"
    )
    parser.add_argument(
        "--collect-all", action="store_true",
        help="Collecte initiale complète (historique max)",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Mise à jour incrémentale",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Scanner les squeezes sur les données collectées",
    )
    parser.add_argument(
        "--hl-only", action="store_true",
        help="Collecter uniquement Hyperliquid",
    )
    parser.add_argument(
        "--sol-only", action="store_true",
        help="Collecter uniquement Solana",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Mode continu (boucle update + scan)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Intervalle en secondes pour le mode live (default: 300)",
    )
    parser.add_argument(
        "--db", type=str, default="squeeze_data.db",
        help="Chemin de la base SQLite",
    )
    parser.add_argument(
        "--min-vol", type=float, default=100_000,
        help="Volume minimum 24h pour Hyperliquid (default: 100K)",
    )
    parser.add_argument(
        "--max-vol", type=float, default=50_000_000,
        help="Volume maximum 24h pour Hyperliquid (default: 50M)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Afficher les stats de la base",
    )

    args = parser.parse_args()

    # Config
    config = CollectorConfig(
        db_path=args.db,
        hl_min_volume_24h=args.min_vol,
        hl_max_volume_24h=args.max_vol,
    )
    collector = SqueezeDataCollector(config)

    if args.stats:
        collector.get_db_stats()
        return

    if args.collect_all:
        if not args.sol_only:
            collector.collect_hyperliquid(full=True)
        if not args.hl_only:
            collector.collect_solana(full=True)
        log.info("✅ Collecte initiale terminée!")
        collector.get_db_stats()

    if args.update:
        if not args.sol_only:
            collector.collect_hyperliquid(full=False)
        if not args.hl_only:
            collector.collect_solana(full=False)
        log.info("✅ Mise à jour terminée!")

    if args.scan:
        collector.scan_squeezes()

    if args.live:
        log.info(f"🔄 Mode live — scan toutes les {args.interval}s")
        while True:
            try:
                if not args.sol_only:
                    collector.collect_hyperliquid(full=False)
                if not args.hl_only:
                    collector.collect_solana(full=False)
                collector.scan_squeezes()
                log.info(f"💤 Prochain scan dans {args.interval}s...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                log.info("Arrêt demandé.")
                break
            except Exception as e:
                log.error(f"Erreur dans la boucle live: {e}")
                time.sleep(60)

    if not any([args.collect_all, args.update, args.scan, args.live, args.stats]):
        parser.print_help()


if __name__ == "__main__":
    main()
