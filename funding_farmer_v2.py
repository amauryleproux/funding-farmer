#!/usr/bin/env python3
"""
Funding Rate Farmer v2 — Hyperliquid
======================================
Bot automatisé avec 3 modes de farming:

1. DELTA-NEUTRAL: Long spot + Short perp sur Hyperliquid (si spot disponible et liquide)
   → Zéro risque directionnel, funding pur
   
2. DIRECTIONNEL: Position perp seule avec SL 5% (si pas de spot)
   → Plus de funding capturé mais exposé au prix

3. ROTATION MULTI-TOKEN: Switch automatique vers le meilleur funding
   → Toujours positionné sur l'opportunité optimale

Logique:
- Scanne tous les tokens, classe par funding absolu
- Pour le meilleur: spot liquide? → delta-neutral, sinon → directionnel
- Réévalue toutes les N minutes, switch si un meilleur token apparaît

Usage:
    python3 funding_farmer_v2.py --dry-run             # Simulation
    python3 funding_farmer_v2.py --live --capital 77    # Live
    python3 funding_farmer_v2.py --live --capital 77 --no-confirm  # Skip confirmation

Prérequis:
    pip install requests python-dotenv eth-account hyperliquid-python-sdk
"""

import os
import sys
import json
import time
import math
import statistics
import requests
import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    import eth_account
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("⚠️  SDK manquant: pip install hyperliquid-python-sdk")

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("farmer_v2")

# ============================================================================
# CONFIG
# ============================================================================

API_URL = "https://api.hyperliquid.xyz/info"

# Spot liquidity minimums
MIN_SPOT_VOLUME_24H = 500       # $500 min volume spot pour delta-neutral
MIN_SPOT_BOOK_DEPTH = 30        # $30 min dans le book (notre taille de position)

@dataclass
class Config:
    secret_key: str = ""
    account_address: str = ""
    mainnet: bool = True
    dry_run: bool = True
    
    # Capital
    capital: float = 77.0
    max_position_pct: float = 0.90    # 90% du capital
    
    # Funding thresholds
    min_funding_pct: float = 0.03     # 0.03%/h minimum pour entrer
    exit_funding_pct: float = 0.005   # Sortir si funding < 0.005%/h
    rotation_advantage_pct: float = 0.03  # Switch si nouveau token a 0.03%/h de plus
    
    # Risk — Directional mode
    stop_loss_pct: float = 5.0        # SL 5% pour directionnel
    
    # Squeeze / Trailing stop
    squeeze_threshold_pct: float = 3.0   # Activer trailing stop à +3% de gain
    trailing_stop_pct: float = 1.2       # Trailing stop: 1.2% sous le peak
    squeeze_check_interval: float = 2.0  # Check prix toutes les 2s en squeeze mode
    
    # Risk — Common
    max_hold_hours: float = 72.0      # Max 3 jours
    min_hold_hours: float = 1.0       # Min 1h avant de considérer rotation
    
    # Filters
    min_volume_24h: float = 100_000   # Perp volume
    min_open_interest: float = 50_000
    max_volatility_24h: float = 8.0   # Max 8% de vol 24h pour directionnel
    min_funding_vol_ratio: float = 0.5  # Funding annualisé / vol annualisée minimum
    blocked_coins: list = field(default_factory=lambda: ["PURR", "HFUN"])
    
    # Operational
    scan_interval: int = 15
    rotation_check_interval: int = 300  # Check rotation toutes les 5min
    
    # Fees
    perp_maker_bps: float = 1.5
    perp_taker_bps: float = 4.5
    spot_maker_bps: float = 4.0
    spot_taker_bps: float = 7.0
    
    @property
    def max_position_usd(self) -> float:
        return self.capital * self.max_position_pct


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SpotMarket:
    coin: str
    spot_coin: str          # ex: "@107" format Hyperliquid
    mid_price: float
    volume_24h: float
    bid_depth_usd: float    # Profondeur bid en USD
    ask_depth_usd: float    # Profondeur ask en USD
    sz_decimals: int = 2


@dataclass 
class Opportunity:
    coin: str
    funding_rate: float         # Horaire décimal
    funding_pct: float          # Horaire %
    annualized_pct: float
    direction: str              # "SHORT" ou "LONG" (direction perp pour collecter)
    mark_price: float
    volume_24h: float
    open_interest: float
    perp_sz_decimals: int
    # Spot info
    has_spot: bool = False
    spot_market: Optional[SpotMarket] = None
    mode: str = "DIRECTIONAL"   # "DELTA_NEUTRAL" ou "DIRECTIONAL"
    # Computed
    hourly_usd_per_1k: float = 0.0
    entry_cost_pct: float = 0.0     # Coût total d'entrée en %
    hours_to_breakeven: float = 999.0
    # Volatility
    volatility_24h: float = 0.0     # Vol 24h en %
    funding_vol_ratio: float = 0.0  # funding annualisé / vol annualisée (plus c'est haut mieux c'est)
    score: float = 0.0              # Score composite final
    # Squeeze indicators
    squeeze_score: float = 0.0      # Score de squeeze 0-100
    funding_accel: float = 0.0      # Accélération du funding (dernières 6h vs 6h avant)
    oi_trend: float = 0.0           # Tendance OI (% change sur 24h)  
    premium_pct: float = 0.0        # Premium mark vs oracle en %


@dataclass
class Position:
    coin: str
    mode: str                   # "DELTA_NEUTRAL" ou "DIRECTIONAL"
    direction: str              # Direction du perp: "SHORT" ou "LONG"
    # Perp leg
    perp_size: float            # Taille perp (négatif si short)
    perp_entry_price: float
    # Spot leg (si delta-neutral)
    spot_size: float = 0.0      # Taille spot
    spot_entry_price: float = 0.0
    spot_coin: str = ""         # "@107" format
    # Tracking
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    funding_at_entry: float = 0.0
    total_funding_collected: float = 0.0
    total_fees: float = 0.0
    status: str = "OPEN"
    # Squeeze tracking
    peak_price: float = 0.0          # Plus haut prix atteint
    peak_pnl_pct: float = 0.0       # Plus haut P&L atteint en %
    squeeze_active: bool = False     # Trailing stop activé?
    trailing_stop_price: float = 0.0 # Prix du trailing stop


@dataclass
class Stats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_trades: int = 0
    delta_neutral_trades: int = 0
    directional_trades: int = 0
    total_funding: float = 0.0
    total_price_pnl: float = 0.0
    total_fees: float = 0.0
    rotations: int = 0
    squeezes_captured: int = 0
    
    @property
    def net_pnl(self) -> float:
        return self.total_funding + self.total_price_pnl - self.total_fees


# ============================================================================
# CLIENT
# ============================================================================

class HLClient:
    def __init__(self, config: Config):
        self.config = config
        self._exchange = None
        self._perp_sz_decimals = {}
        self._spot_markets = {}       # coin -> SpotMarket info
        self._spot_sz_decimals = {}
        
        if HAS_SDK:
            base_url = constants.MAINNET_API_URL if config.mainnet else constants.TESTNET_API_URL
            if config.secret_key and not config.dry_run:
                try:
                    wallet = eth_account.Account.from_key(config.secret_key)
                    self._exchange = Exchange(
                        wallet, base_url,
                        account_address=config.account_address
                    )
                    log.info(f"✅ Client authentifié")
                except Exception as e:
                    log.error(f"❌ Auth échouée: {e}")
        
        self._load_perp_meta()
        self._load_spot_meta()
    
    def _load_perp_meta(self):
        try:
            resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for asset in data[0]["universe"]:
                self._perp_sz_decimals[asset["name"]] = asset["szDecimals"]
            log.info(f"📊 {len(self._perp_sz_decimals)} perp tokens")
        except Exception as e:
            log.error(f"Erreur perp meta: {e}")
    
    def _load_spot_meta(self):
        """Charge les marchés spot et identifie les tokens disponibles."""
        try:
            resp = requests.post(API_URL, json={"type": "spotMetaAndAssetCtxs"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            tokens = data[0]["tokens"]
            universe = data[0]["universe"]
            ctxs = data[1]
            
            spot_count = 0
            for i, market in enumerate(universe):
                try:
                    base_token = tokens[market["tokens"][0]]
                    coin_name = base_token["name"]
                    spot_coin = market["name"]  # ex: "@117"
                    
                    ctx = ctxs[i] if i < len(ctxs) else {}
                    mid_px = float(ctx.get("midPx", "0") or "0")
                    volume = float(ctx.get("dayNtlVlm", "0") or "0")
                    
                    if mid_px > 0 and volume > 0:
                        self._spot_markets[coin_name] = {
                            "spot_coin": spot_coin,
                            "mid_price": mid_px,
                            "volume_24h": volume,
                            "sz_decimals": base_token.get("szDecimals", 2),
                            "token_id": base_token.get("tokenId", 0),
                        }
                        spot_count += 1
                except (KeyError, IndexError, ValueError):
                    continue
            
            log.info(f"📊 {spot_count} spot tokens actifs")
        except Exception as e:
            log.error(f"Erreur spot meta: {e}")
    
    def get_all_funding_rates(self) -> dict:
        try:
            resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = {}
            for asset, ctx in zip(data[0]["universe"], data[1]):
                coin = asset["name"]
                try:
                    mark_px = float(ctx.get("markPx", "0"))
                    if mark_px == 0:
                        continue
                    result[coin] = {
                        "funding": float(ctx.get("funding", "0")),
                        "markPx": mark_px,
                        "oraclePx": float(ctx.get("oraclePx", "0")),
                        "openInterest": float(ctx.get("openInterest", "0")) * mark_px,
                        "volume24h": float(ctx.get("dayNtlVlm", "0")),
                        "szDecimals": asset["szDecimals"],
                    }
                except (ValueError, KeyError):
                    continue
            return result
        except Exception as e:
            log.error(f"Erreur fetch: {e}")
            return {}
    
    def check_spot_liquidity(self, coin: str) -> Optional[SpotMarket]:
        """Vérifie si un token a un spot liquide sur Hyperliquid."""
        spot_info = self._spot_markets.get(coin)
        if not spot_info:
            return None
        
        spot_coin = spot_info["spot_coin"]
        
        # Check volume
        if spot_info["volume_24h"] < MIN_SPOT_VOLUME_24H:
            return None
        
        # Check orderbook depth
        try:
            resp = requests.post(API_URL, json={
                "type": "l2Book", "coin": spot_coin, "nSigFigs": 5
            }, timeout=5)
            resp.raise_for_status()
            book = resp.json()
            
            bids = book.get("levels", [[]])[0]
            asks = book.get("levels", [[]])[1]
            
            bid_depth = sum(float(b["px"]) * float(b["sz"]) for b in bids[:5])
            ask_depth = sum(float(a["px"]) * float(a["sz"]) for a in asks[:5])
            
            mid = float(bids[0]["px"]) if bids else 0
            
            # Vérifier que le prix spot est cohérent avec le perp
            # (éviter les wrapped tokens avec prix différent)
            perp_rates = self.get_all_funding_rates()
            perp_price = perp_rates.get(coin, {}).get("markPx", 0)
            
            if perp_price > 0 and mid > 0:
                price_diff = abs(mid - perp_price) / perp_price
                if price_diff > 0.05:  # > 5% de différence = pas le même token
                    log.debug(f"  {coin}: spot ({mid:.4f}) ≠ perp ({perp_price:.4f}), skip")
                    return None
            
            if bid_depth < MIN_SPOT_BOOK_DEPTH or ask_depth < MIN_SPOT_BOOK_DEPTH:
                return None
            
            return SpotMarket(
                coin=coin,
                spot_coin=spot_coin,
                mid_price=mid,
                volume_24h=spot_info["volume_24h"],
                bid_depth_usd=bid_depth,
                ask_depth_usd=ask_depth,
                sz_decimals=spot_info["sz_decimals"],
            )
            
        except Exception as e:
            log.debug(f"  Erreur book spot {coin}: {e}")
            return None
    
    def get_mid_price(self, coin: str) -> float:
        try:
            resp = requests.post(API_URL, json={"type": "allMids"}, timeout=10)
            resp.raise_for_status()
            return float(resp.json().get(coin, 0))
        except:
            return 0.0
    
    def get_volatility_24h(self, coin: str) -> float:
        """Calcule la volatilité 24h en % à partir des candles 1h."""
        try:
            start = int((time.time() - 24 * 3600) * 1000)
            end = int(time.time() * 1000)
            resp = requests.post(API_URL, json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": "1h", "startTime": start, "endTime": end}
            }, timeout=10)
            resp.raise_for_status()
            candles = resp.json()
            
            if len(candles) < 4:
                return 99.0  # Pas assez de data → considérer comme très volatil
            
            # Calcul des rendements horaires
            returns = []
            for i in range(1, len(candles)):
                close_prev = float(candles[i - 1]["c"])
                close_curr = float(candles[i]["c"])
                if close_prev > 0:
                    returns.append((close_curr - close_prev) / close_prev)
            
            if not returns:
                return 99.0
            
            # Écart-type des rendements × sqrt(24) pour annualiser sur 24h
            std = statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])
            vol_24h = std * math.sqrt(24) * 100  # En %
            
            # Aussi calculer le range haut-bas sur 24h
            high = max(float(c["h"]) for c in candles)
            low = min(float(c["l"]) for c in candles)
            mid = (high + low) / 2
            range_pct = (high - low) / mid * 100 if mid > 0 else 99.0
            
            # Prendre le max des deux mesures (plus conservateur)
            return max(vol_24h, range_pct)
            
        except Exception as e:
            log.debug(f"  Vol {coin}: erreur {e}")
            return 99.0  # En cas d'erreur, considérer très volatil
    
    def get_funding_history(self, coin: str, hours: int = 24) -> list[dict]:
        """Récupère l'historique des funding rates."""
        try:
            start_time = int((time.time() - hours * 3600) * 1000)
            resp = requests.post(API_URL, json={
                "type": "fundingHistory",
                "coin": coin,
                "startTime": start_time
            }, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except:
            return []
    
    def compute_squeeze_score(self, coin: str, funding_rate: float, 
                               mark_price: float, oracle_price: float,
                               open_interest: float, volume_24h: float) -> dict:
        """
        Calcule un score de squeeze (0-100) basé sur 5 indicateurs.
        
        Chaque indicateur est noté de 0 à 20.
        Plus le score est élevé, plus le squeeze est probable.
        
        Ne fonctionne que pour les fundings négatifs (short squeeze potentiel)
        ou positifs (long squeeze potentiel).
        """
        scores = {}
        
        # ─── 1. MAGNITUDE DU FUNDING (0-20) ───
        # Plus le funding est extrême, plus les shorts/longs souffrent
        abs_funding = abs(funding_rate) * 100  # en %
        if abs_funding >= 0.15:
            scores["funding_magnitude"] = 20
        elif abs_funding >= 0.10:
            scores["funding_magnitude"] = 16
        elif abs_funding >= 0.05:
            scores["funding_magnitude"] = 12
        elif abs_funding >= 0.03:
            scores["funding_magnitude"] = 8
        else:
            scores["funding_magnitude"] = 4
        
        # ─── 2. ACCÉLÉRATION DU FUNDING (0-20) ───
        # Funding qui empire = shorts qui s'entassent = bombe qui se charge
        history = self.get_funding_history(coin, hours=12)
        funding_accel = 0.0
        
        if len(history) >= 6:
            # Comparer la moyenne des 6 dernières heures vs les 6 précédentes
            recent = history[-6:]
            older = history[-12:-6] if len(history) >= 12 else history[:len(history)//2]
            
            avg_recent = sum(abs(float(h["fundingRate"])) for h in recent) / len(recent)
            avg_older = sum(abs(float(h["fundingRate"])) for h in older) / len(older)
            
            if avg_older > 0:
                funding_accel = (avg_recent - avg_older) / avg_older * 100  # % d'accélération
            
            # Aussi vérifier la consistance (toutes les heures dans la même direction)
            same_direction = all(
                (float(h["fundingRate"]) < 0) == (funding_rate < 0) 
                for h in recent
            )
            
            if funding_accel > 50 and same_direction:
                scores["funding_accel"] = 20
            elif funding_accel > 20 and same_direction:
                scores["funding_accel"] = 16
            elif funding_accel > 0 and same_direction:
                scores["funding_accel"] = 12
            elif same_direction:
                scores["funding_accel"] = 8  # Stable mais consistant
            else:
                scores["funding_accel"] = 2  # Direction mixte
        else:
            scores["funding_accel"] = 5  # Pas assez de data
        
        # ─── 3. PREMIUM MARK vs ORACLE (0-20) ───
        # Mark < Oracle (funding négatif) = shorts poussent le prix sous la réalité
        # Plus l'écart est grand, plus la correction sera violente
        premium_pct = 0.0
        if oracle_price > 0:
            premium_pct = (mark_price - oracle_price) / oracle_price * 100
        
        # Pour un short squeeze, on veut un premium négatif (mark < oracle)
        if funding_rate < 0:
            abs_premium = abs(min(0, premium_pct))  # Premium négatif seulement
        else:
            abs_premium = abs(max(0, premium_pct))  # Premium positif seulement
        
        if abs_premium >= 1.0:
            scores["premium"] = 20
        elif abs_premium >= 0.5:
            scores["premium"] = 16
        elif abs_premium >= 0.2:
            scores["premium"] = 12
        elif abs_premium >= 0.1:
            scores["premium"] = 8
        else:
            scores["premium"] = 3
        
        # ─── 4. RATIO OI / VOLUME (0-20) ───
        # OI élevé vs volume = beaucoup de positions ouvertes mais peu de trading
        # = positions "coincées" qui devront sortir violemment
        oi_vol_ratio = open_interest / volume_24h if volume_24h > 0 else 0
        
        if oi_vol_ratio >= 0.5:
            scores["oi_concentration"] = 20
        elif oi_vol_ratio >= 0.3:
            scores["oi_concentration"] = 16
        elif oi_vol_ratio >= 0.15:
            scores["oi_concentration"] = 12
        elif oi_vol_ratio >= 0.08:
            scores["oi_concentration"] = 8
        else:
            scores["oi_concentration"] = 3
        
        # ─── 5. CONSISTANCE DIRECTIONNELLE (0-20) ───
        # Si TOUTES les heures récentes sont dans la même direction = pression constante
        if len(history) >= 12:
            all_entries = [float(h["fundingRate"]) for h in history[-12:]]
            same_sign = all(f < 0 for f in all_entries) or all(f > 0 for f in all_entries)
            
            if same_sign:
                avg_rate = sum(abs(f) for f in all_entries) / len(all_entries)
                if avg_rate * 100 >= 0.10:
                    scores["consistency"] = 20
                elif avg_rate * 100 >= 0.05:
                    scores["consistency"] = 16
                else:
                    scores["consistency"] = 12
            else:
                flips = sum(1 for i in range(1, len(all_entries)) if all_entries[i] * all_entries[i-1] < 0)
                if flips <= 1:
                    scores["consistency"] = 10
                elif flips <= 3:
                    scores["consistency"] = 5
                else:
                    scores["consistency"] = 1
        else:
            scores["consistency"] = 5
        
        # ─── TOTAL (avant pénalité) ───
        raw_total = sum(scores.values())
        
        # ─── 6. PÉNALITÉ: SQUEEZE DÉJÀ EU LIEU (-0 à -60) ───
        # Si le prix a déjà bougé massivement dans notre direction,
        # le squeeze est DERRIÈRE nous, pas devant.
        recent_move = self._get_recent_move(coin, hours=4)
        
        # Pour un short squeeze (funding négatif), le move dangereux est UP
        # Pour un long squeeze (funding positif), le move dangereux est DOWN
        if funding_rate < 0:
            already_squeezed_pct = max(0, recent_move)   # Move UP = squeeze déjà fait
        else:
            already_squeezed_pct = max(0, -recent_move)  # Move DOWN = squeeze déjà fait
        
        penalty = 0
        if already_squeezed_pct >= 15:
            penalty = -60  # Move massif = squeeze terminé, ne pas entrer
        elif already_squeezed_pct >= 10:
            penalty = -45
        elif already_squeezed_pct >= 5:
            penalty = -25
        elif already_squeezed_pct >= 3:
            penalty = -10
        
        scores["post_squeeze_penalty"] = penalty
        total = max(0, raw_total + penalty)
        
        return {
            "total": total,
            "raw_total": raw_total,
            "components": scores,
            "funding_accel": funding_accel,
            "premium_pct": premium_pct,
            "oi_vol_ratio": oi_vol_ratio,
            "recent_move_pct": recent_move,
            "already_squeezed_pct": already_squeezed_pct,
        }
    
    def _get_recent_move(self, coin: str, hours: int = 4) -> float:
        """Retourne le % de mouvement de prix sur les N dernières heures.
        Positif = prix a monté, Négatif = prix a baissé.
        Prend le MAX entre le move moyen sur N heures ET la plus grosse bougie 1h récente."""
        try:
            start = int((time.time() - hours * 3600) * 1000)
            end = int(time.time() * 1000)
            resp = requests.post(API_URL, json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": "1h", "startTime": start, "endTime": end}
            }, timeout=10)
            resp.raise_for_status()
            candles = resp.json()
            
            if not candles:
                return 0.0
            
            # Move total sur la période
            open_price = float(candles[0]["o"])
            close_price = float(candles[-1]["c"])
            total_move = 0.0
            if open_price > 0:
                total_move = (close_price - open_price) / open_price * 100
            
            # Plus grosse bougie individuelle (absolue)
            max_candle_move = 0.0
            for c in candles:
                c_open = float(c["o"])
                c_close = float(c["c"])
                c_high = float(c["h"])
                c_low = float(c["l"])
                if c_open > 0:
                    # Body move
                    body = (c_close - c_open) / c_open * 100
                    # Full range (wick to wick)
                    full_range = (c_high - c_low) / c_open * 100
                    
                    candle_move = max(abs(body), full_range)
                    if candle_move > max_candle_move:
                        max_candle_move = candle_move
            
            # Retourner le plus grand des deux (signe = direction du move total)
            sign = 1 if total_move >= 0 else -1
            return sign * max(abs(total_move), max_candle_move)
            
        except:
            return 0.0
    
    def get_account_state(self) -> dict:
        """Retourne l'état complet du compte."""
        try:
            resp = requests.post(API_URL, json={
                "type": "clearinghouseState",
                "user": self.config.account_address
            }, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except:
            return {}
    
    def get_real_positions(self) -> list[dict]:
        """Positions réelles depuis l'exchange."""
        state = self.get_account_state()
        positions = []
        for p in state.get("assetPositions", []):
            pos = p.get("position", {})
            size = float(pos.get("szi", "0"))
            if size != 0:
                positions.append({
                    "coin": pos.get("coin", ""),
                    "size": size,
                    "entryPx": float(pos.get("entryPx", "0")),
                    "unrealizedPnl": float(pos.get("unrealizedPnl", "0")),
                })
        return positions
    
    def get_real_equity(self) -> float:
        state = self.get_account_state()
        return float(state.get("marginSummary", {}).get("accountValue", "0"))
    
    @staticmethod
    def _round_price(price: float, is_buy: bool) -> float:
        if price <= 0:
            return price
        magnitude = math.floor(math.log10(abs(price)))
        decimals = max(0, 4 - magnitude)
        factor = 10 ** decimals
        if is_buy:
            return math.ceil(price * factor) / factor
        else:
            return math.floor(price * factor) / factor
    
    def _place_order(self, coin: str, is_buy: bool, size: float,
                     reduce_only: bool = False, is_spot: bool = False) -> dict:
        """Place un market order (IOC)."""
        if self.config.dry_run or not self._exchange:
            side = "BUY" if is_buy else "SELL"
            mkt = "SPOT" if is_spot else "PERP"
            log.info(f"🧪 [DRY] {side} {size} {coin} {mkt}")
            return {"status": "ok", "dry_run": True, "filled": {"avgPx": "0", "totalSz": str(size)}}
        
        try:
            if is_spot:
                sz_decimals = self._spot_markets.get(coin, {}).get("sz_decimals", 2)
            else:
                sz_decimals = self._perp_sz_decimals.get(coin, 2)
            
            size = round(size, sz_decimals)
            if size == 0:
                return {"status": "error", "msg": "Size too small"}
            
            mid = self.get_mid_price(coin)
            if mid == 0:
                return {"status": "error", "msg": "No mid price"}
            
            slippage = 0.008  # 0.8% slippage 
            price = mid * (1 + slippage) if is_buy else mid * (1 - slippage)
            price = self._round_price(price, is_buy)
            
            log.info(f"  {'SPOT' if is_spot else 'PERP'} {'BUY' if is_buy else 'SELL'} {size} {coin} @ ${price:.6f}")
            
            order_result = self._exchange.order(
                coin, is_buy, size, price,
                {"limit": {"tif": "Ioc"}},
                reduce_only=reduce_only,
            )
            
            log.info(f"  → {order_result}")
            return order_result
            
        except Exception as e:
            log.error(f"❌ Erreur ordre: {e}")
            return {"status": "error", "msg": str(e)}
    
    def place_perp_order(self, coin: str, is_buy: bool, size: float,
                         reduce_only: bool = False) -> dict:
        return self._place_order(coin, is_buy, size, reduce_only, is_spot=False)
    
    def place_spot_order(self, coin: str, is_buy: bool, size: float) -> dict:
        spot_info = self._spot_markets.get(coin)
        if not spot_info:
            return {"status": "error", "msg": f"No spot market for {coin}"}
        spot_coin = spot_info["spot_coin"]
        return self._place_order(spot_coin, is_buy, size, is_spot=True)
    
    @staticmethod
    def parse_order_result(result: dict) -> tuple[bool, float, float]:
        """Parse le résultat d'un ordre. Returns (success, avg_price, filled_size)."""
        if result.get("dry_run"):
            return True, 0.0, float(result.get("filled", {}).get("totalSz", "0"))
        
        if result.get("status") != "ok":
            return False, 0.0, 0.0
        
        response = result.get("response", {})
        data = response.get("data", {})
        statuses = data.get("statuses", [])
        
        if not statuses:
            return False, 0.0, 0.0
        
        first = statuses[0]
        if "filled" in first:
            return True, float(first["filled"]["avgPx"]), float(first["filled"]["totalSz"])
        elif "resting" in first:
            return True, 0.0, 0.0  # Posted but not filled
        else:
            return False, 0.0, 0.0


# ============================================================================
# FARMER V2
# ============================================================================

class FundingFarmerV2:
    def __init__(self, config: Config):
        self.config = config
        self.client = HLClient(config)
        self.position: Optional[Position] = None
        self.stats = Stats()
        self.last_rotation_check = 0.0
        
        # Sync avec l'exchange au démarrage
        self._sync_positions()
    
    def _sync_positions(self):
        """Détecte les positions existantes au démarrage."""
        if self.config.dry_run:
            return
        
        real_positions = self.client.get_real_positions()
        if real_positions:
            pos = real_positions[0]  # Prendre la première
            log.info(f"📡 Position existante détectée: {pos['coin']} size={pos['size']:.4f} @ ${pos['entryPx']:.4f}")
            
            direction = "LONG" if pos["size"] > 0 else "SHORT"
            self.position = Position(
                coin=pos["coin"],
                mode="DIRECTIONAL",  # On assume directionnel au startup
                direction=direction,
                perp_size=pos["size"],
                perp_entry_price=pos["entryPx"],
                entry_time=datetime.now(timezone.utc),  # Approximatif
                status="OPEN",
            )
            log.info(f"  → Reprise position {direction} {pos['coin']}")
        else:
            log.info("📡 Aucune position existante")
    
    def _log_trade(self, action: str, coin: str, details: dict):
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "coin": coin,
            **details,
        }
        with open("trades_v2.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    # ====================================================================
    # SCANNING
    # ====================================================================
    
    def scan_opportunities(self) -> list[Opportunity]:
        """Scanne tous les tokens et classe par squeeze score."""
        all_rates = self.client.get_all_funding_rates()
        candidates = []
        
        # Phase 1: filtrage rapide par funding + volume
        for coin, data in all_rates.items():
            if coin in self.config.blocked_coins:
                continue
            if data["volume24h"] < self.config.min_volume_24h:
                continue
            if data["openInterest"] < self.config.min_open_interest:
                continue
            
            funding = data["funding"]
            abs_funding_pct = abs(funding) * 100
            
            if abs_funding_pct < self.config.min_funding_pct:
                continue
            
            candidates.append((coin, data, funding, abs_funding_pct))
        
        # Phase 2: top 10 par funding pour analyse approfondie
        # (squeeze score requiert des appels API supplémentaires)
        candidates.sort(key=lambda x: x[3], reverse=True)
        candidates = candidates[:10]
        
        opportunities = []
        for coin, data, funding, abs_funding_pct in candidates:
            direction = "SHORT" if funding > 0 else "LONG"
            
            # Calculer volatilité
            vol_24h = self.client.get_volatility_24h(coin)
            
            # Calculer squeeze score
            sq = self.client.compute_squeeze_score(
                coin, funding, data["markPx"], data["oraclePx"],
                data["openInterest"], data["volume24h"]
            )
            squeeze_score = sq["total"]
            
            # Check spot
            spot_market = self.client.check_spot_liquidity(coin)
            has_spot = spot_market is not None
            
            if has_spot:
                mode = "DELTA_NEUTRAL"
                entry_cost = (self.config.spot_taker_bps + self.config.perp_taker_bps) * 2 / 10000 * 100
            else:
                mode = "DIRECTIONAL"
                entry_cost = self.config.perp_taker_bps * 2 / 10000 * 100
                
                # Filtre volatilité SAUF si squeeze score élevé
                # Un token avec vol 15% mais squeeze score 80 = on y va
                if vol_24h > self.config.max_volatility_24h and squeeze_score < 60:
                    log.debug(f"  {coin}: vol {vol_24h:.1f}% high + squeeze {squeeze_score} low, skip")
                    continue
            
            hours_to_be = entry_cost / (abs(funding) * 100) if abs(funding) > 0 else 999
            
            vol_annualized = vol_24h * math.sqrt(365)
            funding_annualized = abs_funding_pct * 24 * 365
            funding_vol_ratio = funding_annualized / vol_annualized if vol_annualized > 0 else 0
            
            # Score final = mélange de funding/vol ratio + squeeze potential
            # Squeeze score (0-100) dominé par le potentiel de squeeze
            if mode == "DELTA_NEUTRAL":
                score = abs_funding_pct * 1000  # DN = priorité funding pur
            else:
                # Score hybride: 40% funding/vol + 60% squeeze
                funding_score = min(abs_funding_pct * funding_vol_ratio * 10, 50)
                score = funding_score * 0.4 + squeeze_score * 0.6
            
            opp = Opportunity(
                coin=coin,
                funding_rate=funding,
                funding_pct=funding * 100,
                annualized_pct=funding_annualized,
                direction=direction,
                mark_price=data["markPx"],
                volume_24h=data["volume24h"],
                open_interest=data["openInterest"],
                perp_sz_decimals=data["szDecimals"],
                has_spot=has_spot,
                spot_market=spot_market,
                mode=mode,
                hourly_usd_per_1k=abs(funding) * 1000,
                entry_cost_pct=entry_cost,
                hours_to_breakeven=hours_to_be,
                volatility_24h=vol_24h,
                funding_vol_ratio=funding_vol_ratio,
                score=score,
                squeeze_score=squeeze_score,
                funding_accel=sq["funding_accel"],
                oi_trend=sq["oi_vol_ratio"],
                premium_pct=sq["premium_pct"],
            )
            opportunities.append(opp)
            
            # Log squeeze details pour les top tokens
            penalty_tag = f" ⚠️MOVED {sq['already_squeezed_pct']:+.1f}%" if sq.get("already_squeezed_pct", 0) > 3 else ""
            log.info(f"  📊 {coin}: fund={abs_funding_pct:.3f}%/h "
                     f"SQ={squeeze_score}/100 (raw:{sq['raw_total']}) "
                     f"prem={sq['premium_pct']:+.2f}% "
                     f"move4h={sq['recent_move_pct']:+.1f}%{penalty_tag} "
                     f"→ score={score:.1f}")
        
        # Trier par score décroissant
        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities
    
    # ====================================================================
    # ENTRY
    # ====================================================================
    
    def enter_position(self, opp: Opportunity):
        """Ouvre une position (delta-neutral ou directionnelle)."""
        coin = opp.coin
        capital = self.config.max_position_usd
        
        if opp.mode == "DELTA_NEUTRAL" and opp.spot_market:
            self._enter_delta_neutral(opp, capital)
        else:
            self._enter_directional(opp, capital)
    
    def _enter_delta_neutral(self, opp: Opportunity, capital: float):
        """Ouvre long spot + short perp (ou inverse)."""
        coin = opp.coin
        spot = opp.spot_market
        half_capital = capital / 2  # Split entre spot et perp
        
        mark_price = opp.mark_price
        size = half_capital / mark_price
        
        log.info(f"\n{'='*60}")
        log.info(f"🛡️  DELTA-NEUTRAL ENTRY: {coin}")
        log.info(f"   Funding: {opp.funding_pct:+.4f}%/h | Mode: {opp.mode}")
        log.info(f"   Capital: ${capital:.2f} (${half_capital:.2f} spot + ${half_capital:.2f} perp)")
        log.info(f"{'='*60}")
        
        # Leg 1: Spot (direction opposée au perp pour hedger)
        # Si funding négatif → perp LONG pour collecter → spot SHORT... 
        # Mais on ne peut pas shorter du spot. 
        # Donc: funding négatif → LONG perp + ... pas possible en spot
        # En fait: funding POSITIF → SHORT perp (collecter) + LONG spot (hedge)
        # funding NÉGATIF → LONG perp (collecter) + SHORT spot... impossible
        
        # Delta-neutral ne marche que quand on SHORT le perp + LONG le spot
        # C'est-à-dire quand le funding est POSITIF (longs paient shorts)
        if opp.direction == "SHORT":
            # Funding positif: short perp + long spot ✓
            spot_is_buy = True
            perp_is_buy = False
        else:
            # Funding négatif: long perp + short spot ✗ (pas de short spot)
            # Fallback en directionnel
            log.info(f"  ⚠️  Funding négatif → pas de short spot possible → fallback directionnel")
            self._enter_directional(opp, capital)
            return
        
        # Place spot order first
        spot_result = self.client.place_spot_order(coin, spot_is_buy, size)
        spot_ok, spot_price, spot_filled = self.client.parse_order_result(spot_result)
        
        if not spot_ok:
            log.error(f"  ❌ Spot order échoué, annulation")
            return
        
        # Place perp order
        perp_result = self.client.place_perp_order(coin, perp_is_buy, size)
        perp_ok, perp_price, perp_filled = self.client.parse_order_result(perp_result)
        
        if not perp_ok:
            log.error(f"  ❌ Perp order échoué, fermeture spot...")
            # Unwind spot
            self.client.place_spot_order(coin, not spot_is_buy, spot_filled)
            return
        
        # Les deux legs sont exécutés
        spot_fee = half_capital * self.config.spot_taker_bps / 10000
        perp_fee = half_capital * self.config.perp_taker_bps / 10000
        total_fees = spot_fee + perp_fee
        
        self.position = Position(
            coin=coin,
            mode="DELTA_NEUTRAL",
            direction=opp.direction,
            perp_size=-size if not perp_is_buy else size,
            perp_entry_price=perp_price or opp.mark_price,
            spot_size=size if spot_is_buy else -size,
            spot_entry_price=spot_price or opp.mark_price,
            spot_coin=spot.spot_coin,
            entry_time=datetime.now(timezone.utc),
            funding_at_entry=opp.funding_rate,
            total_fees=total_fees,
        )
        
        self.stats.total_trades += 1
        self.stats.delta_neutral_trades += 1
        self.stats.total_fees += total_fees
        
        log.info(f"  ✅ DN ouvert: LONG spot {size:.2f} @ ${spot_price:.4f} + SHORT perp @ ${perp_price:.4f}")
        log.info(f"  Fees: ${total_fees:.4f}")
        
        self._log_trade("ENTER_DN", coin, {
            "mode": "DELTA_NEUTRAL",
            "spot_size": size, "spot_price": spot_price,
            "perp_size": size, "perp_price": perp_price,
            "funding_rate": opp.funding_rate, "fees": total_fees,
        })
    
    def _enter_directional(self, opp: Opportunity, capital: float):
        """Ouvre une position directionnelle."""
        coin = opp.coin
        mark_price = opp.mark_price
        size = capital / mark_price
        size = round(size, opp.perp_sz_decimals)
        
        if size == 0:
            log.warning(f"  {coin}: taille trop petite")
            return
        
        is_buy = opp.direction == "LONG"
        
        log.info(f"\n{'='*60}")
        log.info(f"⚡ DIRECTIONAL ENTRY: {opp.direction} {size} {coin}")
        log.info(f"   Funding: {opp.funding_pct:+.4f}%/h | SL: {self.config.stop_loss_pct}%")
        log.info(f"   Squeeze Score: {opp.squeeze_score:.0f}/100 | Premium: {opp.premium_pct:+.2f}%")
        log.info(f"   Position: ~${capital:.2f}")
        log.info(f"{'='*60}")
        
        result = self.client.place_perp_order(coin, is_buy, size)
        ok, price, filled = self.client.parse_order_result(result)
        
        if not ok:
            log.error(f"  ❌ Ordre rejeté")
            return
        
        fee = capital * self.config.perp_taker_bps / 10000
        
        self.position = Position(
            coin=coin,
            mode="DIRECTIONAL",
            direction=opp.direction,
            perp_size=filled if is_buy else -filled,
            perp_entry_price=price or mark_price,
            entry_time=datetime.now(timezone.utc),
            funding_at_entry=opp.funding_rate,
            total_fees=fee,
        )
        
        self.stats.total_trades += 1
        self.stats.directional_trades += 1
        self.stats.total_fees += fee
        
        log.info(f"  ✅ Rempli @ ${price:.4f}")
        
        self._log_trade("ENTER_DIR", coin, {
            "mode": "DIRECTIONAL", "direction": opp.direction,
            "size": filled, "price": price or mark_price,
            "funding_rate": opp.funding_rate, "fee": fee,
            "squeeze_score": opp.squeeze_score, "premium_pct": opp.premium_pct,
        })
    
    # ====================================================================
    # EXIT
    # ====================================================================
    
    def close_position(self, reason: str):
        """Ferme la position actuelle (les deux legs si DN)."""
        pos = self.position
        if not pos:
            return
        
        log.info(f"\n{'='*60}")
        log.info(f"🚪 EXIT: {pos.coin} | Mode: {pos.mode} | Raison: {reason}")
        
        current_price = self.client.get_mid_price(pos.coin)
        
        # Fermer le perp
        perp_close = self.client.place_perp_order(
            pos.coin, 
            pos.perp_size < 0,  # Buy to close short, sell to close long
            abs(pos.perp_size),
            reduce_only=True
        )
        perp_ok, perp_exit_price, _ = self.client.parse_order_result(perp_close)
        perp_exit_price = perp_exit_price or current_price
        
        # Calculer P&L perp
        if pos.perp_size > 0:  # Long
            perp_pnl = (perp_exit_price - pos.perp_entry_price) * abs(pos.perp_size)
        else:  # Short
            perp_pnl = (pos.perp_entry_price - perp_exit_price) * abs(pos.perp_size)
        
        exit_fee = abs(pos.perp_size) * perp_exit_price * self.config.perp_taker_bps / 10000
        
        # Fermer le spot si delta-neutral
        spot_pnl = 0.0
        if pos.mode == "DELTA_NEUTRAL" and pos.spot_size != 0:
            spot_close = self.client.place_spot_order(
                pos.coin,
                pos.spot_size < 0,  # Buy to close short
                abs(pos.spot_size)
            )
            spot_ok, spot_exit_price, _ = self.client.parse_order_result(spot_close)
            spot_exit_price = spot_exit_price or current_price
            
            if pos.spot_size > 0:
                spot_pnl = (spot_exit_price - pos.spot_entry_price) * abs(pos.spot_size)
            else:
                spot_pnl = (pos.spot_entry_price - spot_exit_price) * abs(pos.spot_size)
            
            exit_fee += abs(pos.spot_size) * spot_exit_price * self.config.spot_taker_bps / 10000
        
        # Estimer le funding collecté
        hold_hours = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
        estimated_funding = abs(pos.funding_at_entry) * abs(pos.perp_size) * pos.perp_entry_price * hold_hours
        
        total_pnl = perp_pnl + spot_pnl
        pos.total_fees += exit_fee
        net = total_pnl + estimated_funding - pos.total_fees
        
        log.info(f"   Durée: {hold_hours:.1f}h")
        log.info(f"   P&L prix: ${perp_pnl:+.4f} (perp) + ${spot_pnl:+.4f} (spot)")
        log.info(f"   Funding estimé: +${estimated_funding:.4f}")
        log.info(f"   Fees totales: -${pos.total_fees:.4f}")
        log.info(f"   ═══ NET: ${net:+.4f} ═══")
        log.info(f"{'='*60}")
        
        # Update stats
        self.stats.total_price_pnl += total_pnl
        self.stats.total_funding += estimated_funding
        self.stats.total_fees += exit_fee
        
        self._log_trade("EXIT", pos.coin, {
            "mode": pos.mode, "reason": reason,
            "perp_pnl": perp_pnl, "spot_pnl": spot_pnl,
            "funding": estimated_funding, "fees": pos.total_fees,
            "net": net, "hold_hours": hold_hours,
        })
        
        self.position = None
    
    # ====================================================================
    # CHECK EXITS & ROTATION
    # ====================================================================
    
    def check_price_fast(self) -> Optional[str]:
        """Check rapide du prix uniquement (pas de funding/rotation).
        Utilisé en mode squeeze pour réagir vite."""
        pos = self.position
        if not pos or pos.mode == "DELTA_NEUTRAL":
            return None
        
        current_price = self.client.get_mid_price(pos.coin)
        if current_price == 0:
            return None
        
        # P&L
        if pos.perp_size > 0:
            pnl_pct = (current_price - pos.perp_entry_price) / pos.perp_entry_price * 100
        else:
            pnl_pct = (pos.perp_entry_price - current_price) / pos.perp_entry_price * 100
        
        position_usd = abs(pos.perp_size) * pos.perp_entry_price
        pnl_usd = pnl_pct / 100 * position_usd
        
        # Mettre à jour le peak
        if pnl_pct > pos.peak_pnl_pct:
            pos.peak_pnl_pct = pnl_pct
            pos.peak_price = current_price
            
            if pos.squeeze_active:
                # Recalculer le trailing stop
                if pos.perp_size > 0:  # Long
                    pos.trailing_stop_price = pos.peak_price * (1 - self.config.trailing_stop_pct / 100)
                else:  # Short
                    pos.trailing_stop_price = pos.peak_price * (1 + self.config.trailing_stop_pct / 100)
                
                log.info(f"  🔺 Nouveau peak: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | "
                         f"TS: ${pos.trailing_stop_price:.4f}")
        
        # Détecter l'entrée en squeeze mode
        if not pos.squeeze_active and pnl_pct >= self.config.squeeze_threshold_pct:
            pos.squeeze_active = True
            pos.peak_price = current_price
            pos.peak_pnl_pct = pnl_pct
            
            if pos.perp_size > 0:
                pos.trailing_stop_price = current_price * (1 - self.config.trailing_stop_pct / 100)
            else:
                pos.trailing_stop_price = current_price * (1 + self.config.trailing_stop_pct / 100)
            
            log.info(f"\n  🚀🚀🚀 SQUEEZE DÉTECTÉ sur {pos.coin}!")
            log.info(f"  PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            log.info(f"  Trailing stop activé @ ${pos.trailing_stop_price:.4f} "
                     f"({self.config.trailing_stop_pct}% sous le peak)")
            log.info(f"  Mode rapide: check toutes les {self.config.squeeze_check_interval}s")
        
        # Check trailing stop
        if pos.squeeze_active:
            hit = False
            if pos.perp_size > 0 and current_price <= pos.trailing_stop_price:
                hit = True
            elif pos.perp_size < 0 and current_price >= pos.trailing_stop_price:
                hit = True
            
            if hit:
                self.stats.squeezes_captured += 1
                return (f"TRAILING_STOP 🎯 (peak: {pos.peak_pnl_pct:+.1f}%, "
                        f"exit: {pnl_pct:+.1f}%, captured: ${pnl_usd:+.2f})")
            
            # Log squeeze status
            distance_to_ts = abs(current_price - pos.trailing_stop_price) / current_price * 100
            log.info(f"  🚀 SQUEEZE {pos.coin}: PnL {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | "
                     f"Peak: {pos.peak_pnl_pct:+.2f}% | TS: ${pos.trailing_stop_price:.4f} "
                     f"({distance_to_ts:.2f}% away)")
        
        # Stop loss classique (toujours actif)
        hold_hours = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
        if pnl_pct < -self.config.stop_loss_pct and hold_hours > 0.1:
            return f"STOP_LOSS ({pnl_pct:+.1f}%)"
        
        return None
    
    def check_exit_conditions(self, current_rates: dict) -> Optional[str]:
        """Check complet: prix + funding + durée. Appelé toutes les N secondes."""
        pos = self.position
        if not pos:
            return None
        
        now = datetime.now(timezone.utc)
        hold_hours = (now - pos.entry_time).total_seconds() / 3600
        current_price = self.client.get_mid_price(pos.coin)
        
        if current_price == 0:
            return None
        
        # P&L directionnel
        if pos.perp_size > 0:
            pnl_pct = (current_price - pos.perp_entry_price) / pos.perp_entry_price * 100
        else:
            pnl_pct = (pos.perp_entry_price - current_price) / pos.perp_entry_price * 100
        
        position_usd = abs(pos.perp_size) * pos.perp_entry_price
        pnl_usd = pnl_pct / 100 * position_usd
        
        # Funding estimé
        estimated_funding = abs(pos.funding_at_entry) * abs(pos.perp_size) * pos.perp_entry_price * hold_hours
        pos.total_funding_collected = estimated_funding
        
        # Mettre à jour le peak (aussi dans le check complet)
        if pnl_pct > pos.peak_pnl_pct:
            pos.peak_pnl_pct = pnl_pct
            pos.peak_price = current_price
            if pos.squeeze_active:
                if pos.perp_size > 0:
                    pos.trailing_stop_price = pos.peak_price * (1 - self.config.trailing_stop_pct / 100)
                else:
                    pos.trailing_stop_price = pos.peak_price * (1 + self.config.trailing_stop_pct / 100)
        
        # Détecter squeeze
        if not pos.squeeze_active and pnl_pct >= self.config.squeeze_threshold_pct:
            pos.squeeze_active = True
            pos.peak_price = current_price
            pos.peak_pnl_pct = pnl_pct
            if pos.perp_size > 0:
                pos.trailing_stop_price = current_price * (1 - self.config.trailing_stop_pct / 100)
            else:
                pos.trailing_stop_price = current_price * (1 + self.config.trailing_stop_pct / 100)
            log.info(f"\n  🚀🚀🚀 SQUEEZE DÉTECTÉ sur {pos.coin}! PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            log.info(f"  Trailing stop @ ${pos.trailing_stop_price:.4f}")
        
        # 1. Trailing stop (si squeeze actif)
        if pos.squeeze_active:
            hit = False
            if pos.perp_size > 0 and current_price <= pos.trailing_stop_price:
                hit = True
            elif pos.perp_size < 0 and current_price >= pos.trailing_stop_price:
                hit = True
            if hit:
                self.stats.squeezes_captured += 1
                return (f"TRAILING_STOP 🎯 (peak: {pos.peak_pnl_pct:+.1f}%, "
                        f"exit: {pnl_pct:+.1f}%, captured: ${pnl_usd:+.2f})")
        
        # 2. STOP LOSS — uniquement en mode directionnel
        if pos.mode == "DIRECTIONAL":
            if pnl_pct < -self.config.stop_loss_pct and hold_hours > 0.1:
                return f"STOP_LOSS ({pnl_pct:+.1f}%)"
        
        # 3. Durée max
        if hold_hours >= self.config.max_hold_hours:
            return f"MAX_HOLD ({hold_hours:.0f}h)"
        
        # 4. Funding flip ou disparition (pas pendant un squeeze)
        if not pos.squeeze_active:
            coin_data = current_rates.get(pos.coin, {})
            current_funding = coin_data.get("funding", 0)
            current_funding_pct = abs(current_funding) * 100
            
            if hold_hours >= self.config.min_hold_hours:
                if current_funding_pct < self.config.exit_funding_pct:
                    return f"LOW_FUNDING ({current_funding_pct:.4f}%/h)"
                if pos.direction == "SHORT" and current_funding < 0:
                    return "FUNDING_FLIP"
                elif pos.direction == "LONG" and current_funding > 0:
                    return "FUNDING_FLIP"
        
        # Log status
        hourly_funding = abs(current_rates.get(pos.coin, {}).get("funding", 0)) * position_usd
        net = pnl_usd + estimated_funding - pos.total_fees
        
        mode_icon = "🛡️" if pos.mode == "DELTA_NEUTRAL" else "⚡"
        squeeze_tag = " 🚀SQUEEZE" if pos.squeeze_active else ""
        log.info(
            f"  {mode_icon} {pos.coin}: {pos.direction} {pos.mode}{squeeze_tag} | "
            f"{hold_hours:.1f}h | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | "
            f"Fund: +${estimated_funding:.4f} (${hourly_funding:.4f}/h) | "
            f"Net: ${net:+.4f}"
        )
        
        return None
    
    def should_rotate(self, opportunities: list[Opportunity]) -> Optional[Opportunity]:
        """Vérifie si on devrait switch vers un meilleur token."""
        if not self.position:
            return None
        
        pos = self.position
        hold_hours = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
        
        # Ne pas rotater trop tôt (min hold)
        if hold_hours < self.config.min_hold_hours:
            return None
        
        # Ne pas checker la rotation trop souvent
        now = time.time()
        if now - self.last_rotation_check < self.config.rotation_check_interval:
            return None
        self.last_rotation_check = now
        
        # Funding actuel de notre position
        current_rates = self.client.get_all_funding_rates()
        current_funding = abs(current_rates.get(pos.coin, {}).get("funding", 0)) * 100
        
        for opp in opportunities:
            if opp.coin == pos.coin:
                continue
            
            abs_opp_funding = abs(opp.funding_pct)
            advantage = abs_opp_funding - current_funding
            
            # Le nouveau token doit avoir un avantage significatif
            # pour couvrir les fees de rotation
            if advantage >= self.config.rotation_advantage_pct:
                # Bonus si le nouveau token supporte delta-neutral
                if opp.has_spot and pos.mode == "DIRECTIONAL":
                    log.info(f"  🔄 ROTATION: {pos.coin} ({current_funding:.4f}%/h) → "
                             f"{opp.coin} ({abs_opp_funding:.4f}%/h) [DN available!]")
                    return opp
                elif advantage >= self.config.rotation_advantage_pct:
                    log.info(f"  🔄 ROTATION: {pos.coin} ({current_funding:.4f}%/h) → "
                             f"{opp.coin} ({abs_opp_funding:.4f}%/h) [+{advantage:.4f}%/h]")
                    return opp
        
        return None
    
    # ====================================================================
    # DISPLAY
    # ====================================================================
    
    def display_status(self, opportunities: list[Opportunity]):
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        mode = "🧪 DRY RUN" if self.config.dry_run else "🔴 LIVE"
        
        print(f"\n┌─ {mode} ─ Funding Farmer v2 ─ {now} {'─'*15}┐")
        
        # Top opportunities
        if opportunities:
            print(f"│ 🔍 Top tokens (trié par squeeze score):")
            for opp in opportunities[:5]:
                mode_tag = "🛡️DN" if opp.has_spot else "⚡DIR"
                sq_bar = "🔥" if opp.squeeze_score >= 70 else ("⭐" if opp.squeeze_score >= 50 else "  ")
                move_warn = f"⚠️{opp.oi_trend:+.0f}%" if hasattr(opp, '_recent_move') else ""
                print(f"│  {sq_bar} {opp.coin:<8} {opp.funding_pct:>+.4f}%/h "
                      f"SQ:{opp.squeeze_score:>3.0f}/100 "
                      f"prem:{opp.premium_pct:>+.2f}% "
                      f"{mode_tag}")
        
        # Position actuelle
        print(f"│")
        if self.position:
            pos = self.position
            hold_h = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
            mode_icon = "🛡️" if pos.mode == "DELTA_NEUTRAL" else "⚡"
            pos_usd = abs(pos.perp_size) * pos.perp_entry_price
            print(f"│ {mode_icon} Position: {pos.direction} {abs(pos.perp_size):.2f} {pos.coin} "
                  f"(~${pos_usd:.0f}) | {pos.mode}")
            print(f"│   Entrée: ${pos.perp_entry_price:.4f} | Hold: {hold_h:.1f}h | "
                  f"Fund: +${pos.total_funding_collected:.4f}")
            if pos.squeeze_active:
                print(f"│   🚀 SQUEEZE ACTIF | Peak: {pos.peak_pnl_pct:+.1f}% | "
                      f"TS: ${pos.trailing_stop_price:.4f} | Check: {self.config.squeeze_check_interval}s")
            elif pos.mode == "DIRECTIONAL":
                print(f"│   SL: {self.config.stop_loss_pct}% | "
                      f"Squeeze trigger: +{self.config.squeeze_threshold_pct}%")
        else:
            print(f"│ 📊 Aucune position")
        
        # Stats
        print(f"│")
        print(f"│ 📈 Total: {self.stats.total_trades} trades "
              f"({self.stats.delta_neutral_trades} DN / {self.stats.directional_trades} DIR) "
              f"| {self.stats.rotations} rot | {self.stats.squeezes_captured} 🚀")
        print(f"│   Fund: +${self.stats.total_funding:.4f} | "
              f"Prix: ${self.stats.total_price_pnl:+.4f} | "
              f"Fees: -${self.stats.total_fees:.4f} | "
              f"Net: ${self.stats.net_pnl:+.4f}")
        print(f"└{'─'*62}┘")
    
    # ====================================================================
    # MAIN LOOP
    # ====================================================================
    
    def run(self):
        log.info(f"🚀 Funding Farmer v2")
        log.info(f"   Mode: {'DRY RUN' if self.config.dry_run else 'LIVE'}")
        log.info(f"   Capital: ${self.config.capital}")
        log.info(f"   Stratégie: Delta-Neutral si spot dispo, sinon Directionnel (SL {self.config.stop_loss_pct}%)")
        log.info(f"   Filtre vol: max {self.config.max_volatility_24h}% vol 24h pour directionnel")
        log.info(f"   Squeeze: trailing stop {self.config.trailing_stop_pct}% activé à +{self.config.squeeze_threshold_pct}%")
        log.info(f"   Scoring: Squeeze Score 0-100 (funding mag + accel + premium + OI/vol + consistance)")
        log.info(f"   Rotation: switch si avantage > {self.config.rotation_advantage_pct}%/h")
        log.info(f"   Seuil entrée: {self.config.min_funding_pct}%/h | Sortie: {self.config.exit_funding_pct}%/h")
        log.info(f"   Intervalles: normal={self.config.scan_interval}s | squeeze={self.config.squeeze_check_interval}s")
        print()
        
        cycle = 0
        last_full_scan = 0
        
        try:
            while True:
                cycle += 1
                now = time.time()
                
                is_squeeze = self.position and self.position.squeeze_active
                
                # ─── FAST PATH: squeeze mode → check prix seulement ───
                if is_squeeze:
                    exit_reason = self.check_price_fast()
                    if exit_reason:
                        self.close_position(exit_reason)
                        # Après un squeeze exit, re-scanner immédiatement
                        last_full_scan = 0
                    
                    # Full scan moins souvent en squeeze (toutes les 30s)
                    if now - last_full_scan >= 30:
                        current_rates = self.client.get_all_funding_rates()
                        exit_reason = self.check_exit_conditions(current_rates)
                        if exit_reason:
                            self.close_position(exit_reason)
                        last_full_scan = now
                    
                    time.sleep(self.config.squeeze_check_interval)
                    continue
                
                # ─── NORMAL PATH: scan complet ───
                
                # 1. Scanner (pas à chaque cycle, c'est lourd avec la vol)
                opportunities = []
                if now - last_full_scan >= self.config.scan_interval:
                    opportunities = self.scan_opportunities()
                    last_full_scan = now
                
                # 2. Check exits (prix + funding + squeeze detection)
                current_rates = self.client.get_all_funding_rates()
                if self.position:
                    exit_reason = self.check_exit_conditions(current_rates)
                    if exit_reason:
                        self.close_position(exit_reason)
                
                # 3. Check rotation
                if self.position and opportunities:
                    better = self.should_rotate(opportunities)
                    if better:
                        self.close_position(f"ROTATION → {better.coin}")
                        self.stats.rotations += 1
                        time.sleep(1)
                        self.enter_position(better)
                
                # 4. Entrer si pas de position
                if not self.position and opportunities:
                    best = opportunities[0]
                    log.info(f"  🎯 Best: {best.coin} {best.funding_pct:+.4f}%/h "
                             f"SQ:{best.squeeze_score:.0f}/100 "
                             f"vol:{best.volatility_24h:.1f}% ({best.mode})")
                    self.enter_position(best)
                
                # 5. Display
                if cycle % 10 == 1:
                    self.display_status(opportunities)
                
                # Normal speed quand pas de squeeze
                time.sleep(self.config.scan_interval)
                
        except KeyboardInterrupt:
            log.info("\n⏹ Arrêt")
            self.display_status([])
            
            if self.position and not self.config.dry_run:
                print(f"\n⚠️  Position ouverte: {self.position.coin} ({self.position.mode})")
                resp = input("Fermer ? (y/n): ")
                if resp.lower() == "y":
                    self.close_position("MANUAL_STOP")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Funding Farmer v2 — Hyperliquid")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-confirm", action="store_true", help="Skip live confirmation")
    parser.add_argument("--capital", type=float, default=77.0)
    parser.add_argument("--threshold", type=float, default=0.03, help="Min funding %%/h")
    parser.add_argument("--stop-loss", type=float, default=5.0, help="SL %% directionnel")
    parser.add_argument("--squeeze-trigger", type=float, default=3.0, help="Activer trailing stop à +N%%")
    parser.add_argument("--trailing-stop", type=float, default=1.2, help="Trailing stop: N%% sous le peak")
    parser.add_argument("--squeeze-interval", type=float, default=2.0, help="Check interval en squeeze (secondes)")
    parser.add_argument("--rotation-adv", type=float, default=0.03, help="Avantage min pour rotation")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--max-vol", type=float, default=8.0, help="Max volatilité 24h %% pour directionnel")
    parser.add_argument("--max-hold", type=float, default=72.0)
    
    args = parser.parse_args()
    
    config = Config(
        secret_key=os.getenv("HL_SECRET_KEY", ""),
        account_address=os.getenv("HL_ACCOUNT_ADDRESS", ""),
        dry_run=not args.live,
        capital=args.capital,
        min_funding_pct=args.threshold,
        stop_loss_pct=args.stop_loss,
        squeeze_threshold_pct=args.squeeze_trigger,
        trailing_stop_pct=args.trailing_stop,
        squeeze_check_interval=args.squeeze_interval,
        rotation_advantage_pct=args.rotation_adv,
        max_volatility_24h=args.max_vol,
        scan_interval=args.interval,
        max_hold_hours=args.max_hold,
    )
    
    if not config.dry_run:
        print("╔══════════════════════════════════════════════════╗")
        print("║  🔴 LIVE TRADING — v2 Delta-Neutral + Rotation  ║")
        print("╚══════════════════════════════════════════════════╝")
        
        if not config.secret_key or not config.account_address:
            print("❌ HL_SECRET_KEY / HL_ACCOUNT_ADDRESS manquant")
            sys.exit(1)
        
        if not args.no_confirm:
            confirm = input("\n⚠️  Confirmer live ? (yes/no): ")
            if confirm.lower() != "yes":
                sys.exit(0)
    else:
        print("╔══════════════════════════════════════════════════╗")
        print("║  🧪 DRY RUN — v2 Delta-Neutral + Rotation       ║")
        print("╚══════════════════════════════════════════════════╝")
    
    farmer = FundingFarmerV2(config)
    farmer.run()


if __name__ == "__main__":
    main()