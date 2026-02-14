"""
=============================================================================
SQUEEZE AUTO-TRADER — Bot 100% automatique
=============================================================================
Pipeline complet :
  1. Collecte les données OHLCV (Hyperliquid + Solana)
  2. Détecte les squeezes en temps réel
  3. Entre automatiquement (Phase 1: directionnel)
  4. Gère les positions (stops, trailing, target)
  5. Transition Phase 2 (funding farming) post-breakout
  6. Log tout dans SQLite

Usage:
  # Dry-run d'abord (TOUJOURS)
  python squeeze_auto_trader.py --dry-run

  # Live avec petit capital
  python squeeze_auto_trader.py --live --size 30

  # Live avec alertes console uniquement (pas de trades)
  python squeeze_auto_trader.py --alert-only
=============================================================================
"""

import argparse
import json
import logging
import sqlite3
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Local imports
from squeeze_detector import (
    SqueezeDetector, SqueezeConfig, SqueezeSignal,
    SqueezePhase, BreakoutDirection,
)
from squeeze_data_collector import (
    CollectorConfig, HyperliquidFetcher, SolanaFetcher,
    init_db, save_candles, load_candles_df, save_token_meta,
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("squeeze_trader")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TraderConfig:
    """Configuration du trader automatique."""

    # --- Mode ---
    dry_run: bool = True                  # TOUJOURS commencer en dry-run
    alert_only: bool = False              # Affiche les signaux sans trader

    # --- Capital & Sizing ---
    max_position_usd: float = 30.0        # Taille max par position
    max_positions: int = 2                 # Max positions simultanées
    max_total_exposure_usd: float = 80.0  # Exposition totale max
    leverage: float = 3.0                 # Levier

    # --- Entry Conditions ---
    min_squeeze_score: float = 0.55       # Score minimum pour entrer
    min_direction_confidence: float = 0.6 # Confiance direction minimum
    min_volume_ratio: float = 0.3         # Volume ratio minimum
    required_phases: list[str] = field(
        default_factory=lambda: ["ready", "firing"]
    )
    # Cooldown: pas re-entrer sur le même token pendant N minutes
    entry_cooldown_minutes: float = 60.0

    # --- Exit / Risk Management ---
    stop_loss_atr_mult: float = 1.5       # Stop = entry ± 1.5 × ATR
    take_profit_atr_mult: float = 3.0     # TP = entry ± 3.0 × ATR
    trailing_stop_pct: float = 0.015      # Trailing stop à 1.5%
    trailing_activation_pct: float = 0.01 # Active trailing après +1%
    max_holding_hours: float = 24.0       # Fermer après 24h max
    check_interval_sec: float = 30.0      # Vérifier positions toutes les 30s

    # --- Scan Frequency ---
    scan_interval_sec: float = 300.0      # Scanner les squeezes toutes les 5 min
    data_update_interval_sec: float = 900.0  # Update données toutes les 15 min

    # --- Hyperliquid ---
    hl_min_volume: float = 100_000        # Volume min 24h
    hl_max_volume: float = 50_000_000     # Volume max 24h

    # --- Safety ---
    max_daily_loss_usd: float = 15.0      # Stop trading si -$15 dans la journée
    max_trades_per_day: int = 10          # Max trades par jour
    cooldown_after_loss_sec: float = 300.0  # 5 min cooldown après une perte

    # --- Database ---
    db_path: str = "squeeze_data.db"


# =============================================================================
# POSITION TRACKER
# =============================================================================

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class LivePosition:
    """Position live avec gestion des stops."""
    coin: str
    side: PositionSide
    entry_price: float
    size: float                    # En unités du coin
    size_usd: float
    entry_time: float
    stop_price: float
    take_profit_price: float
    trailing_stop_price: float = 0.0
    trailing_activated: bool = False
    highest_price: float = 0.0     # Pour trailing (long)
    lowest_price: float = float('inf')  # Pour trailing (short)
    squeeze_score: float = 0.0
    pattern: str = ""              # "squeeze_ready", "squeeze_firing"
    atr_at_entry: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    def update_trailing(self, current_price: float, config: TraderConfig):
        """Met à jour le trailing stop."""
        if self.is_long:
            self.highest_price = max(self.highest_price, current_price)
            profit_pct = (current_price - self.entry_price) / self.entry_price

            if profit_pct >= config.trailing_activation_pct:
                self.trailing_activated = True
                new_trailing = self.highest_price * (1 - config.trailing_stop_pct)
                self.trailing_stop_price = max(
                    self.trailing_stop_price, new_trailing
                )
        else:
            self.lowest_price = min(self.lowest_price, current_price)
            profit_pct = (self.entry_price - current_price) / self.entry_price

            if profit_pct >= config.trailing_activation_pct:
                self.trailing_activated = True
                new_trailing = self.lowest_price * (1 + config.trailing_stop_pct)
                if self.trailing_stop_price == 0:
                    self.trailing_stop_price = new_trailing
                else:
                    self.trailing_stop_price = min(
                        self.trailing_stop_price, new_trailing
                    )

    def should_exit(self, current_price: float, config: TraderConfig) -> tuple[bool, str]:
        """Vérifie si on doit sortir."""
        # Stop loss
        if self.is_long and current_price <= self.stop_price:
            return True, "stop_loss"
        if not self.is_long and current_price >= self.stop_price:
            return True, "stop_loss"

        # Take profit
        if self.is_long and current_price >= self.take_profit_price:
            return True, "take_profit"
        if not self.is_long and current_price <= self.take_profit_price:
            return True, "take_profit"

        # Trailing stop
        if self.trailing_activated:
            if self.is_long and current_price <= self.trailing_stop_price:
                return True, "trailing_stop"
            if not self.is_long and current_price >= self.trailing_stop_price:
                return True, "trailing_stop"

        # Max holding time
        elapsed_hours = (time.time() - self.entry_time) / 3600
        if elapsed_hours >= config.max_holding_hours:
            return True, "max_time"

        return False, ""


# =============================================================================
# HYPERLIQUID EXECUTOR (compatible avec ton client existant)
# =============================================================================

class HyperliquidExecutor:
    """
    Exécuteur d'ordres Hyperliquid.
    Utilise directement le SDK hyperliquid-python.
    """

    def __init__(self, secret_key: str = "", account_address: str = "", dry_run: bool = True):
        self.dry_run = dry_run
        self._client = None
        self._exchange = None
        self._info = None
        self._sz_decimals: dict[str, int] = {}
        self._price_decimals: dict[str, int] = {}
        self._meta_loaded = False

        if not dry_run and secret_key:
            try:
                import eth_account
                from hyperliquid.info import Info
                from hyperliquid.exchange import Exchange
                from hyperliquid.utils import constants

                self._account = eth_account.Account.from_key(secret_key)
                self._info = Info(constants.MAINNET_API_URL, skip_ws=True)
                self._exchange = Exchange(
                    self._account,
                    constants.MAINNET_API_URL,
                    account_address=account_address or None,
                )
                self._account_address = account_address or self._account.address
                log.info(f"✅ Connecté à Hyperliquid: {self._account_address[:10]}...")
            except Exception as e:
                log.error(f"❌ Erreur connexion HL: {e}")
                log.warning("Passage en mode dry-run")
                self.dry_run = True
        else:
            self._info_session = requests.Session()
            self._info_session.headers["Content-Type"] = "application/json"

    def _load_meta(self):
        """Charge les métadonnées des assets."""
        if self._meta_loaded:
            return
        try:
            if self._info:
                meta = self._info.meta()
            else:
                resp = self._info_session.post(
                    "https://api.hyperliquid.xyz/info",
                    json={"type": "meta"},
                    timeout=10,
                )
                meta = resp.json()

            for asset in meta.get("universe", []):
                name = asset.get("name", "")
                self._sz_decimals[name] = asset.get("szDecimals", 2)
                # Tick size: derive price decimals from the asset
                # Hyperliquid uses "significantFigures" or we infer from price
                self._price_decimals[name] = self._get_price_decimals(asset)
            self._meta_loaded = True
        except Exception as e:
            log.error(f"Erreur chargement meta: {e}")

    @staticmethod
    def _get_price_decimals(asset: dict) -> int:
        """Determine price decimals from asset metadata."""
        # Hyperliquid tick sizes vary per asset
        # We use significant figures approach based on typical price ranges
        # Most assets: 5 significant figures for price
        # This will be refined with actual mid prices
        return 6  # Safe default, will be refined in round_price

    def get_sz_decimals(self, coin: str) -> int:
        self._load_meta()
        return self._sz_decimals.get(coin, 2)

    def round_price(self, coin: str, price: float) -> float:
        """
        Round price to valid tick size for Hyperliquid.
        Rules: max 5 significant figures, and integer if price >= 100K
        """
        if price <= 0:
            return price

        # Hyperliquid rule: 5 significant figures
        if price >= 100_000:
            return round(price)
        elif price >= 10_000:
            return round(price, 1)
        elif price >= 1_000:
            return round(price, 2)
        elif price >= 100:
            return round(price, 3)
        elif price >= 10:
            return round(price, 4)
        elif price >= 1:
            return round(price, 5)
        else:
            # For sub-$1 prices, find appropriate decimals
            # e.g., 0.046784 → 5 sig figs → 0.046784 is 5 sig figs
            import math
            if price > 0:
                magnitude = math.floor(math.log10(abs(price)))
                decimals = 5 - 1 - magnitude  # 5 sig figs
                decimals = max(0, min(8, decimals))
                return round(price, decimals)
            return price

    def round_size(self, coin: str, size: float) -> float:
        dec = self.get_sz_decimals(coin)
        return round(size, dec)

    def get_mid_price(self, coin: str) -> Optional[float]:
        """Prix mid actuel."""
        try:
            if self._info:
                mids = self._info.all_mids()
            else:
                resp = self._info_session.post(
                    "https://api.hyperliquid.xyz/info",
                    json={"type": "allMids"},
                    timeout=10,
                )
                mids = resp.json()
            return float(mids.get(coin, 0))
        except Exception as e:
            log.error(f"Erreur mid price {coin}: {e}")
            return None

    def get_account_equity(self) -> float:
        """Retourne l'equity du compte."""
        if self.dry_run:
            return 100.0  # Simulated
        try:
            state = self._info.user_state(self._account_address)
            return float(state.get("marginSummary", {}).get("accountValue", 0))
        except:
            return 0.0

    def get_open_positions(self) -> list[dict]:
        """Retourne les positions ouvertes."""
        if self.dry_run:
            return []
        try:
            state = self._info.user_state(self._account_address)
            positions = []
            for p in state.get("assetPositions", []):
                pos = p.get("position", {})
                size = float(pos.get("szi", 0))
                if abs(size) > 0:
                    positions.append({
                        "coin": pos.get("coin", ""),
                        "size": size,
                        "entry_px": float(pos.get("entryPx", 0)),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    })
            return positions
        except Exception as e:
            log.error(f"Erreur positions: {e}")
            return []

    def place_market_order(
        self, coin: str, is_buy: bool, size_usd: float, current_price: float
    ) -> dict:
        """
        Place un ordre market (limite agressive pour fill immédiat).
        Retourne {"success": bool, "fill_price": float, "size": float}
        """
        size = self.round_size(coin, size_usd / current_price)

        if size <= 0:
            return {"success": False, "error": "size_zero"}

        # Prix agressif pour fill immédiat (±0.5% du mid)
        # Round to valid tick size for Hyperliquid
        slippage = 0.005
        if is_buy:
            limit_price = self.round_price(coin, current_price * (1 + slippage))
        else:
            limit_price = self.round_price(coin, current_price * (1 - slippage))

        side_str = "BUY" if is_buy else "SELL"
        log.info(f"📤 ORDER | {side_str} {coin} | size={size} (~${size_usd:.0f}) | px={limit_price}")

        if self.dry_run:
            log.info(f"  🔵 DRY RUN — ordre simulé")
            return {
                "success": True,
                "fill_price": current_price,
                "size": size,
                "dry_run": True,
            }

        try:
            result = self._exchange.order(
                coin, is_buy, size, limit_price,
                {"limit": {"tif": "Ioc"}},  # Immediate-or-Cancel
                reduce_only=False,
            )

            if result["status"] == "ok":
                statuses = result["response"]["data"]["statuses"]
                if statuses:
                    s = statuses[0]
                    if "filled" in s:
                        return {
                            "success": True,
                            "fill_price": float(s["filled"]["avgPx"]),
                            "size": float(s["filled"]["totalSz"]),
                        }
                    elif "resting" in s:
                        return {
                            "success": True,
                            "fill_price": limit_price,
                            "size": size,
                        }
                    elif "error" in s:
                        return {"success": False, "error": s["error"]}

            return {"success": False, "error": str(result)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def close_position(self, coin: str, size: float, is_long: bool, current_price: float) -> dict:
        """Ferme une position."""
        return self.place_market_order(
            coin, is_buy=not is_long,
            size_usd=abs(size * current_price),
            current_price=current_price,
        )


# =============================================================================
# TRADE LOGGER
# =============================================================================

def init_trade_log(conn: sqlite3.Connection):
    """Table pour logger les trades."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            coin        TEXT NOT NULL,
            side        TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price  REAL,
            size        REAL NOT NULL,
            size_usd    REAL NOT NULL,
            pnl         REAL DEFAULT 0,
            pnl_pct     REAL DEFAULT 0,
            entry_time  INTEGER NOT NULL,
            exit_time   INTEGER,
            exit_reason TEXT,
            squeeze_score REAL,
            pattern     TEXT,
            status      TEXT DEFAULT 'open'
        );
    """)
    conn.commit()


# =============================================================================
# MAIN BOT
# =============================================================================

class SqueezeAutoTrader:
    """Bot de trading automatique basé sur les squeezes."""

    def __init__(
        self,
        config: TraderConfig,
        secret_key: str = "",
        account_address: str = "",
    ):
        self.config = config
        self.conn = init_db(config.db_path)
        init_trade_log(self.conn)

        # Composants
        self.executor = HyperliquidExecutor(
            secret_key=secret_key,
            account_address=account_address,
            dry_run=config.dry_run,
        )
        self.detector = SqueezeDetector()
        self.hl_fetcher = HyperliquidFetcher(CollectorConfig(
            db_path=config.db_path,
            hl_min_volume_24h=config.hl_min_volume,
            hl_max_volume_24h=config.hl_max_volume,
        ))

        # State
        self.positions: dict[str, LivePosition] = {}  # coin → position
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_reset_time: float = time.time()
        self.last_entry_time: dict[str, float] = {}  # coin → timestamp
        self.last_loss_time: float = 0.0
        self.last_data_update: float = 0.0
        self.last_scan_time: float = 0.0

        # Tokens à scanner (mis à jour dynamiquement)
        self.watchlist: list[dict] = []

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Boucle principale du bot."""
        mode = "DRY-RUN" if self.config.dry_run else "🔴 LIVE"
        if self.config.alert_only:
            mode = "ALERT-ONLY"

        print("=" * 70)
        print(f"  SQUEEZE AUTO-TRADER | Mode: {mode}")
        print(f"  Position size: ${self.config.max_position_usd}")
        print(f"  Max positions: {self.config.max_positions}")
        print(f"  Leverage: {self.config.leverage}x")
        print(f"  Stop: {self.config.stop_loss_atr_mult}×ATR | "
              f"TP: {self.config.take_profit_atr_mult}×ATR | "
              f"Trailing: {self.config.trailing_stop_pct:.1%}")
        print(f"  Max daily loss: ${self.config.max_daily_loss_usd}")
        print("=" * 70)

        if not self.config.dry_run and not self.config.alert_only:
            equity = self.executor.get_account_equity()
            print(f"  Account equity: ${equity:.2f}")
            existing = self.executor.get_open_positions()
            if existing:
                print(f"  ⚠️  Positions existantes: {len(existing)}")
                for p in existing:
                    print(f"    {p['coin']}: size={p['size']}, uPnL=${p['unrealized_pnl']:.2f}")
            print("=" * 70)

        print("\n🚀 Démarrage...\n")

        try:
            while True:
                self._reset_daily_if_needed()
                self._check_safety()

                now = time.time()

                # 1. Update données périodique
                if now - self.last_data_update >= self.config.data_update_interval_sec:
                    self._update_data()
                    self.last_data_update = now

                # 2. Scanner les squeezes
                if now - self.last_scan_time >= self.config.scan_interval_sec:
                    signals = self._scan_squeezes()
                    self._process_signals(signals)
                    self.last_scan_time = now

                # 3. Gérer les positions ouvertes
                self._manage_positions()

                # Sleep
                time.sleep(self.config.check_interval_sec)

        except KeyboardInterrupt:
            log.info("⛔ Arrêt demandé.")
            self._close_all_positions("manual_stop")
            self._print_summary()

    # =========================================================================
    # DATA UPDATE
    # =========================================================================

    def _update_data(self):
        """Met à jour les données OHLCV (incrémental, rate-limit safe)."""
        log.info("📊 Mise à jour des données...")

        try:
            # Lister les tokens
            tokens = self.hl_fetcher.get_all_tokens()
            self.watchlist = tokens

            # Sauvegarder meta
            for t in tokens:
                save_token_meta(self.conn, t)

            # Candles : seulement les dernières 12h (pas l'historique complet)
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - (12 * 3600 * 1000)  # 12h back

            updated = 0
            errors = 0
            for i, token in enumerate(tokens):
                coin = token["symbol"]
                try:
                    candles = self.hl_fetcher.fetch_candles(
                        coin, "1h", start_time=start_ms, end_time=now_ms
                    )
                    if candles:
                        save_candles(self.conn, "hyperliquid", coin, coin, "1h", candles)
                        updated += 1
                except Exception as e:
                    errors += 1
                    if "429" in str(e):
                        # Rate limited — back off progressively
                        wait = min(10, 2 + errors * 0.5)
                        log.warning(f"  Rate limited, waiting {wait:.0f}s...")
                        time.sleep(wait)

                # 1s entre chaque requête (safe pour HL)
                time.sleep(1.0)

                # Log progress tous les 50 tokens
                if (i + 1) % 50 == 0:
                    log.info(f"  Progress: {i+1}/{len(tokens)} ({updated} OK, {errors} erreurs)")

            log.info(f"📊 {updated}/{len(tokens)} tokens mis à jour ({errors} erreurs)")

        except Exception as e:
            log.error(f"Erreur update données: {e}")

    # =========================================================================
    # SQUEEZE SCANNING
    # =========================================================================

    def _scan_squeezes(self) -> list[SqueezeSignal]:
        """Scanne tous les tokens pour détecter les squeezes."""
        signals = []

        # Funding rates
        try:
            funding_rates = self.hl_fetcher.fetch_funding_rates()
        except:
            funding_rates = {}

        # Scanner chaque token
        tokens = self.conn.execute(
            """SELECT DISTINCT symbol, display_name FROM candles
               WHERE source = 'hyperliquid'
               GROUP BY symbol HAVING COUNT(*) >= 100"""
        ).fetchall()

        for symbol, display_name in tokens:
            df = load_candles_df(self.conn, "hyperliquid", symbol, "1h", limit=300)
            if df.empty or len(df) < 100:
                continue

            try:
                signal = self.detector.analyze(
                    df, symbol,
                    funding_rate=funding_rates.get(symbol, 0.0),
                )
                if signal.phase != SqueezePhase.NO_SQUEEZE:
                    signals.append(signal)
            except:
                pass

        signals.sort(key=lambda s: s.score, reverse=True)

        # Log le dashboard rapide
        actionable = [
            s for s in signals
            if s.phase.value in self.config.required_phases
            and s.direction != BreakoutDirection.UNKNOWN
            and s.direction_confidence >= self.config.min_direction_confidence
            and s.score >= self.config.min_squeeze_score
            and s.volume_ratio >= self.config.min_volume_ratio
        ]

        if actionable:
            log.info(f"🔍 {len(actionable)} signaux actionables:")
            for s in actionable[:5]:
                log.info(
                    f"  {s.coin:<12} {s.phase.value:<8} "
                    f"score={s.score:.2f} dir={s.direction.value} "
                    f"conf={s.direction_confidence:.0%} "
                    f"move={s.expected_move_pct:.1%}"
                )

        return signals

    # =========================================================================
    # SIGNAL PROCESSING → TRADE ENTRY
    # =========================================================================

    def _process_signals(self, signals: list[SqueezeSignal]):
        """Traite les signaux et entre en position si conditions remplies."""
        if self.config.alert_only:
            return

        for signal in signals:
            # Filtres
            if signal.phase.value not in self.config.required_phases:
                continue
            if signal.score < self.config.min_squeeze_score:
                continue
            if signal.direction == BreakoutDirection.UNKNOWN:
                continue
            if signal.direction_confidence < self.config.min_direction_confidence:
                continue
            if signal.volume_ratio < self.config.min_volume_ratio:
                continue

            coin = signal.coin

            # Déjà en position sur ce coin ?
            if coin in self.positions:
                continue

            # Max positions atteint ?
            if len(self.positions) >= self.config.max_positions:
                continue

            # Cooldown sur ce coin ?
            last_entry = self.last_entry_time.get(coin, 0)
            if time.time() - last_entry < self.config.entry_cooldown_minutes * 60:
                continue

            # Cooldown après perte ?
            if self.last_loss_time > 0:
                if time.time() - self.last_loss_time < self.config.cooldown_after_loss_sec:
                    continue

            # Daily limits ?
            if self.daily_trades >= self.config.max_trades_per_day:
                continue
            if self.daily_pnl <= -self.config.max_daily_loss_usd:
                continue

            # Exposition totale ?
            current_exposure = sum(p.size_usd for p in self.positions.values())
            if current_exposure + self.config.max_position_usd > self.config.max_total_exposure_usd:
                continue

            # ✅ ENTRER
            self._enter_position(signal)

    def _enter_position(self, signal: SqueezeSignal):
        """Ouvre une position basée sur un signal de squeeze."""
        coin = signal.coin
        is_long = signal.direction == BreakoutDirection.LONG

        # Prix actuel
        price = self.executor.get_mid_price(coin)
        if not price or price <= 0:
            log.warning(f"Impossible d'obtenir le prix pour {coin}")
            return

        # Calculer stops basés sur l'ATR
        atr = signal.atr_value
        if atr <= 0:
            atr = price * 0.02  # Fallback 2%

        if is_long:
            stop_price = price - self.config.stop_loss_atr_mult * atr
            tp_price = price + self.config.take_profit_atr_mult * atr
        else:
            stop_price = price + self.config.stop_loss_atr_mult * atr
            tp_price = price - self.config.take_profit_atr_mult * atr

        size_usd = self.config.max_position_usd * self.config.leverage

        log.info("=" * 60)
        log.info(f"🎯 SIGNAL DÉTECTÉ — {coin}")
        log.info(f"  Phase: {signal.phase.value} | Score: {signal.score:.2f}")
        log.info(f"  Direction: {'LONG 📈' if is_long else 'SHORT 📉'} "
                 f"(conf: {signal.direction_confidence:.0%})")
        log.info(f"  Prix: {price} | ATR: {atr:.4f}")
        log.info(f"  Stop: {stop_price:.4f} | TP: {tp_price:.4f}")
        log.info(f"  Size: ${size_usd:.0f} ({self.config.leverage}x)")
        log.info(f"  Expected move: {signal.expected_move_pct:.1%}")
        log.info("=" * 60)

        # Exécuter l'ordre
        result = self.executor.place_market_order(
            coin, is_buy=is_long,
            size_usd=size_usd,
            current_price=price,
        )

        if result.get("success"):
            fill_price = result.get("fill_price", price)
            fill_size = result.get("size", size_usd / price)

            position = LivePosition(
                coin=coin,
                side=PositionSide.LONG if is_long else PositionSide.SHORT,
                entry_price=fill_price,
                size=fill_size,
                size_usd=size_usd,
                entry_time=time.time(),
                stop_price=stop_price,
                take_profit_price=tp_price,
                highest_price=fill_price,
                lowest_price=fill_price,
                squeeze_score=signal.score,
                pattern=f"squeeze_{signal.phase.value}",
                atr_at_entry=atr,
            )

            self.positions[coin] = position
            self.last_entry_time[coin] = time.time()
            self.daily_trades += 1

            # Log dans la DB
            self.conn.execute(
                """INSERT INTO trades
                   (coin, side, entry_price, size, size_usd,
                    entry_time, squeeze_score, pattern, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                (coin, "long" if is_long else "short", fill_price,
                 fill_size, size_usd, int(time.time()),
                 signal.score, f"squeeze_{signal.phase.value}"),
            )
            self.conn.commit()

            log.info(f"✅ POSITION OUVERTE | {coin} {'LONG' if is_long else 'SHORT'} "
                     f"@ {fill_price}")
        else:
            log.error(f"❌ Ordre échoué pour {coin}: {result.get('error', 'unknown')}")

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def _manage_positions(self):
        """Gère toutes les positions ouvertes."""
        for coin in list(self.positions.keys()):
            pos = self.positions[coin]

            # Prix actuel
            price = self.executor.get_mid_price(coin)
            if not price:
                continue

            # Calculer P&L
            if pos.is_long:
                pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price * pos.size_usd
            else:
                pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price * pos.size_usd

            # Update trailing stop
            pos.update_trailing(price, self.config)

            # Check exit
            should_exit, reason = pos.should_exit(price, self.config)

            if should_exit:
                self._exit_position(coin, price, reason)

    def _exit_position(self, coin: str, exit_price: float, reason: str):
        """Ferme une position."""
        pos = self.positions[coin]

        # P&L final
        if pos.is_long:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usd = pnl_pct * pos.size_usd
        # Fees estimées
        fees = pos.size_usd * 0.00035 * 2  # Entry + exit taker fees
        net_pnl = pnl_usd - fees

        emoji = "🟢" if net_pnl > 0 else "🔴"
        holding_min = (time.time() - pos.entry_time) / 60

        log.info("=" * 60)
        log.info(f"{emoji} POSITION FERMÉE | {coin}")
        log.info(f"  Raison: {reason}")
        log.info(f"  Entry: {pos.entry_price} → Exit: {exit_price}")
        log.info(f"  PnL: ${net_pnl:+.2f} ({pnl_pct:+.2%}) | Fees: ${fees:.2f}")
        log.info(f"  Durée: {holding_min:.0f} min | Score: {pos.squeeze_score:.2f}")
        log.info("=" * 60)

        # Exécuter la fermeture
        result = self.executor.close_position(
            coin, pos.size, pos.is_long, exit_price
        )

        # Update state
        self.daily_pnl += net_pnl
        if net_pnl < 0:
            self.last_loss_time = time.time()

        # Log dans la DB
        self.conn.execute(
            """UPDATE trades SET
                exit_price = ?, exit_time = ?, exit_reason = ?,
                pnl = ?, pnl_pct = ?, status = 'closed'
               WHERE coin = ? AND status = 'open'""",
            (exit_price, int(time.time()), reason,
             net_pnl, pnl_pct, coin),
        )
        self.conn.commit()

        del self.positions[coin]

        log.info(f"📊 Daily PnL: ${self.daily_pnl:+.2f} | "
                 f"Trades: {self.daily_trades} | "
                 f"Open positions: {len(self.positions)}")

    def _close_all_positions(self, reason: str):
        """Ferme toutes les positions."""
        for coin in list(self.positions.keys()):
            price = self.executor.get_mid_price(coin)
            if price:
                self._exit_position(coin, price, reason)

    # =========================================================================
    # SAFETY & UTILS
    # =========================================================================

    def _check_safety(self):
        """Vérifie les limites de sécurité."""
        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            if self.positions:
                log.warning(f"⚠️ Daily loss limit (${self.daily_pnl:.2f}), fermeture des positions")
                self._close_all_positions("daily_loss_limit")

    def _reset_daily_if_needed(self):
        """Reset les compteurs journaliers à minuit UTC."""
        now = time.time()
        if now - self.daily_reset_time > 86400:
            log.info(f"📅 Reset journalier | PnL hier: ${self.daily_pnl:+.2f}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_reset_time = now

    def _print_summary(self):
        """Affiche le résumé final."""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DE SESSION")
        print("=" * 60)

        trades = self.conn.execute(
            """SELECT side, entry_price, exit_price, pnl, pnl_pct,
                      exit_reason, coin, squeeze_score
               FROM trades WHERE status = 'closed'
               ORDER BY exit_time DESC LIMIT 20"""
        ).fetchall()

        if trades:
            total_pnl = sum(t[3] for t in trades if t[3])
            wins = sum(1 for t in trades if t[3] and t[3] > 0)
            losses = len(trades) - wins

            print(f"  Total trades: {len(trades)}")
            print(f"  Wins: {wins} | Losses: {losses}")
            print(f"  Win rate: {wins/len(trades)*100:.0f}%")
            print(f"  Total PnL: ${total_pnl:+.2f}")

            print(f"\n  Derniers trades:")
            for t in trades[:10]:
                emoji = "🟢" if t[3] and t[3] > 0 else "🔴"
                print(f"    {emoji} {t[6]:<10} {t[0]:>5} | "
                      f"PnL ${t[3]:+.2f} ({t[4]:+.1%}) | {t[5]}")
        else:
            print("  Aucun trade exécuté.")

        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Squeeze Auto-Trader")

    # Mode
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Mode simulation (default)")
    parser.add_argument("--live", action="store_true",
                        help="Mode live trading")
    parser.add_argument("--alert-only", action="store_true",
                        help="Afficher les signaux sans trader")

    # Credentials (ou via env vars)
    parser.add_argument("--secret-key", type=str, default="",
                        help="Clé privée Hyperliquid (ou env HL_SECRET_KEY)")
    parser.add_argument("--account", type=str, default="",
                        help="Adresse du compte (ou env HL_ACCOUNT)")

    # Sizing
    parser.add_argument("--size", type=float, default=30.0,
                        help="Taille par position en USD (default: 30)")
    parser.add_argument("--max-pos", type=int, default=2,
                        help="Max positions simultanées (default: 2)")
    parser.add_argument("--leverage", type=float, default=3.0,
                        help="Levier (default: 3)")

    # Risk
    parser.add_argument("--max-daily-loss", type=float, default=15.0,
                        help="Perte max journalière en USD (default: 15)")
    parser.add_argument("--stop-atr", type=float, default=1.5,
                        help="Stop loss en multiple d'ATR (default: 1.5)")
    parser.add_argument("--tp-atr", type=float, default=3.0,
                        help="Take profit en multiple d'ATR (default: 3.0)")

    # Timing
    parser.add_argument("--scan-interval", type=int, default=300,
                        help="Intervalle de scan en secondes (default: 300)")

    # Data
    parser.add_argument("--db", type=str, default="squeeze_data.db",
                        help="Base SQLite")
    parser.add_argument("--min-vol", type=float, default=100_000)
    parser.add_argument("--max-vol", type=float, default=50_000_000)

    args = parser.parse_args()

    # Résoudre le mode
    import os
    dry_run = not args.live
    secret_key = args.secret_key or os.environ.get("HL_SECRET_KEY", "")
    account = args.account or os.environ.get("HL_ACCOUNT", "")

    if args.live and not secret_key:
        print("❌ Mode live requiert --secret-key ou env HL_SECRET_KEY")
        sys.exit(1)

    config = TraderConfig(
        dry_run=dry_run,
        alert_only=args.alert_only,
        max_position_usd=args.size,
        max_positions=args.max_pos,
        leverage=args.leverage,
        max_daily_loss_usd=args.max_daily_loss,
        stop_loss_atr_mult=args.stop_atr,
        take_profit_atr_mult=args.tp_atr,
        scan_interval_sec=args.scan_interval,
        db_path=args.db,
        hl_min_volume=args.min_vol,
        hl_max_volume=args.max_vol,
    )

    bot = SqueezeAutoTrader(config, secret_key=secret_key, account_address=account)
    bot.run()


if __name__ == "__main__":
    main()
