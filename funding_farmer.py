#!/usr/bin/env python3
"""
Funding Rate Farmer — Hyperliquid
===================================
Bot automatisé qui ouvre des positions pour capturer les funding rates élevés.

STRATÉGIE:
- Scanne les funding rates toutes les minutes
- Quand un token a un funding > seuil → ouvre une position dans la direction payée
  (funding positif → SHORT, funding négatif → LONG)
- Maintient la position tant que le funding reste élevé
- Ferme quand: funding tombe sous le seuil, stop-loss atteint, ou durée max

RISQUES (mode sans hedge):
- Exposition directionnelle → le prix peut bouger contre nous
- Le stop-loss protège, mais réduit les gains cumulés
- Les fees d'entrée/sortie doivent être couvertes par le funding collecté

Usage:
    python funding_farmer.py --dry-run                    # Simulation
    python funding_farmer.py                              # Live trading  
    python funding_farmer.py --capital 110 --max-pos 50   # Custom params
    python funding_farmer.py --coins ETH,SOL              # Tokens spécifiques

Prérequis:
    pip install requests python-dotenv eth-account hyperliquid-python-sdk
"""

import os
import sys
import json
import time
import hmac
import hashlib
import requests
import argparse
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# Essayer d'importer dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Essayer d'importer le SDK Hyperliquid
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    import eth_account
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("⚠️  SDK Hyperliquid non trouvé. Install: pip install hyperliquid-python-sdk")
    print("   Le bot fonctionnera en mode scanner-only.\n")


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("funding_farmer")


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class FarmerConfig:
    # API
    secret_key: str = ""
    account_address: str = ""
    mainnet: bool = True
    
    # Stratégie
    min_funding_pct: float = 0.015    # 0.015%/h minimum pour entrer (1.5 bps)
    exit_funding_pct: float = 0.005   # 0.005%/h — fermer si funding tombe en dessous
    
    # Risk management
    capital: float = 110.0            # Capital total
    max_position_pct: float = 0.40    # Max 40% du capital par position
    max_positions: int = 3            # Max 3 positions simultanées
    max_leverage: float = 3.0         # Levier max
    stop_loss_pct: float = 0.8        # Stop loss à 0.8% de mouvement adverse
    take_profit_pct: float = 2.0      # Take profit optionnel à 2%
    min_hold_hours: float = 2.0       # Durée minimum (couvrir les fees)
    max_hold_hours: float = 48.0      # Durée max d'une position
    
    # Filtres
    min_volume_24h: float = 100_000   # Volume minimum $100K
    min_open_interest: float = 50_000  # OI minimum $50K
    allowed_coins: list = field(default_factory=list)  # Vide = tous les coins
    blocked_coins: list = field(default_factory=lambda: ["PURR", "HFUN"])  # Coins à éviter
    
    # Opérationnel
    scan_interval: int = 60           # Scan toutes les 60s
    dry_run: bool = True              # Mode simulation par défaut
    log_file: str = "funding_farmer.log"
    
    # Fees Hyperliquid (sans VIP)
    maker_fee_bps: float = 1.5
    taker_fee_bps: float = 4.5
    
    @property
    def max_position_usd(self) -> float:
        return self.capital * self.max_position_pct
    
    @property  
    def roundtrip_fee_pct(self) -> float:
        """Coût aller-retour en % (2 × maker fee si limit orders)."""
        return self.maker_fee_bps * 2 / 100  # = 0.03%


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Position:
    coin: str
    direction: str          # "SHORT" ou "LONG"
    size: float             # Taille en unités du token
    entry_price: float
    entry_time: datetime
    funding_at_entry: float  # Funding rate au moment de l'entrée
    total_funding_collected: float = 0.0
    funding_payments: int = 0
    unrealized_pnl: float = 0.0
    status: str = "OPEN"    # OPEN, CLOSED
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    fees_paid: float = 0.0


@dataclass  
class FarmStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_positions: int = 0
    winning_positions: int = 0
    total_funding_collected: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return (self.winning_positions / self.total_positions * 100) if self.total_positions > 0 else 0
    
    @property
    def net_pnl(self) -> float:
        return self.total_pnl + self.total_funding_collected - self.total_fees


# ============================================================================
# HYPERLIQUID CLIENT (simplifié, self-contained)
# ============================================================================

API_URL = "https://api.hyperliquid.xyz/info"


class HLClient:
    """Client Hyperliquid simplifié pour le funding farmer."""
    
    def __init__(self, config: FarmerConfig):
        self.config = config
        self._exchange = None
        self._info = None
        self._meta = None
        self._sz_decimals = {}
        
        base_url = constants.MAINNET_API_URL if config.mainnet else constants.TESTNET_API_URL
        
        if HAS_SDK:
            self._info = Info(base_url, skip_ws=True)
            
            if config.secret_key and not config.dry_run:
                try:
                    wallet = eth_account.Account.from_key(config.secret_key)
                    self._exchange = Exchange(
                        wallet, base_url,
                        account_address=config.account_address
                    )
                    log.info(f"✅ Client authentifié (mainnet={config.mainnet})")
                except Exception as e:
                    log.error(f"❌ Auth échouée: {e}")
        
        self._load_meta()
    
    def _load_meta(self):
        """Charge les métadonnées."""
        try:
            resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._meta = data[0]["universe"]
            self._ctxs = data[1]
            
            for asset in self._meta:
                self._sz_decimals[asset["name"]] = asset["szDecimals"]
            
            log.info(f"📊 {len(self._meta)} tokens chargés")
        except Exception as e:
            log.error(f"Erreur chargement meta: {e}")
            self._meta = []
            self._ctxs = []
    
    def get_all_funding_rates(self) -> dict:
        """Retourne {coin: {funding, markPx, oraclePx, openInterest, volume}}."""
        try:
            resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            meta = data[0]["universe"]
            ctxs = data[1]
            
            result = {}
            for asset, ctx in zip(meta, ctxs):
                coin = asset["name"]
                try:
                    mark_px = float(ctx.get("markPx", "0"))
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
            log.error(f"Erreur fetch funding: {e}")
            return {}
    
    def get_mid_price(self, coin: str) -> float:
        """Prix mid d'un token."""
        try:
            resp = requests.post(API_URL, json={"type": "allMids"}, timeout=10)
            resp.raise_for_status()
            mids = resp.json()
            return float(mids.get(coin, 0))
        except:
            return 0.0
    
    def get_account_equity(self) -> float:
        """Equity du compte."""
        if not self.config.account_address:
            return self.config.capital
        try:
            resp = requests.post(API_URL, json={
                "type": "clearinghouseState",
                "user": self.config.account_address
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("marginSummary", {}).get("accountValue", "0"))
        except:
            return self.config.capital
    
    def get_positions(self) -> list[dict]:
        """Positions actuelles."""
        if not self.config.account_address:
            return []
        try:
            resp = requests.post(API_URL, json={
                "type": "clearinghouseState",
                "user": self.config.account_address
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            positions = []
            for pos in data.get("assetPositions", []):
                p = pos.get("position", {})
                size = float(p.get("szi", "0"))
                if size != 0:
                    positions.append({
                        "coin": p.get("coin", ""),
                        "size": size,
                        "entryPx": float(p.get("entryPx", "0")),
                        "unrealizedPnl": float(p.get("unrealizedPnl", "0")),
                        "liquidationPx": float(p.get("liquidationPx", "0") or "0"),
                    })
            return positions
        except:
            return []
    
    def place_order(self, coin: str, is_buy: bool, size: float, 
                    price: Optional[float] = None, reduce_only: bool = False) -> dict:
        """Place un ordre limit (ALO/post-only) ou market."""
        if self.config.dry_run or not self._exchange:
            side = "BUY" if is_buy else "SELL"
            log.info(f"🧪 [DRY RUN] {side} {size} {coin} @ {price or 'MARKET'}")
            return {"status": "ok", "dry_run": True}
        
        try:
            sz_decimals = self._sz_decimals.get(coin, 2)
            size = round(size, sz_decimals)
            
            if size == 0:
                return {"status": "error", "msg": "Size too small"}
            
            if price:
                # Limit order, post-only (ALO) pour payer maker fee
                order_result = self._exchange.order(
                    coin, is_buy, size, price,
                    {"limit": {"tif": "Alo"}},
                    reduce_only=reduce_only,
                )
            else:
                # Market order (IOC aggressif)
                # On utilise un prix très favorable pour simuler un market order
                mid = self.get_mid_price(coin)
                slippage = 0.002  # 0.2% slippage
                aggressive_price = mid * (1 + slippage) if is_buy else mid * (1 - slippage)
                aggressive_price = round(aggressive_price, 6)
                
                order_result = self._exchange.order(
                    coin, is_buy, size, aggressive_price,
                    {"limit": {"tif": "Ioc"}},
                    reduce_only=reduce_only,
                )
            
            log.info(f"📋 Ordre placé: {'BUY' if is_buy else 'SELL'} {size} {coin} → {order_result}")
            return order_result
            
        except Exception as e:
            log.error(f"❌ Erreur ordre: {e}")
            return {"status": "error", "msg": str(e)}
    
    def close_position(self, coin: str, size: float) -> dict:
        """Ferme une position (market order, reduce_only)."""
        # Si size > 0 on est long → sell pour fermer
        # Si size < 0 on est short → buy pour fermer
        is_buy = size < 0
        abs_size = abs(size)
        return self.place_order(coin, is_buy, abs_size, price=None, reduce_only=True)


# ============================================================================
# FUNDING FARMER
# ============================================================================

class FundingFarmer:
    """Bot de farming de funding rates."""
    
    def __init__(self, config: FarmerConfig):
        self.config = config
        self.client = HLClient(config)
        self.positions: dict[str, Position] = {}  # coin -> Position
        self.closed_positions: list[Position] = []
        self.stats = FarmStats()
        self.trade_log: list[dict] = []
        
        # Charger l'état précédent si existe
        self._load_state()
    
    def _save_state(self):
        """Sauvegarde l'état du bot."""
        state = {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "stats": asdict(self.stats),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        # Convertir datetimes
        for k, v in state["positions"].items():
            v["entry_time"] = v["entry_time"].isoformat() if isinstance(v["entry_time"], datetime) else v["entry_time"]
            if v.get("exit_time"):
                v["exit_time"] = v["exit_time"].isoformat() if isinstance(v["exit_time"], datetime) else v["exit_time"]
        state["stats"]["started_at"] = state["stats"]["started_at"].isoformat() if isinstance(state["stats"]["started_at"], datetime) else state["stats"]["started_at"]
        
        with open("farmer_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_state(self):
        """Charge l'état précédent."""
        if Path("farmer_state.json").exists():
            try:
                with open("farmer_state.json") as f:
                    state = json.load(f)
                log.info("📂 État précédent chargé")
                # TODO: reconstituer les positions si besoin
            except:
                pass
    
    def _log_trade(self, action: str, coin: str, details: dict):
        """Log un trade."""
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "coin": coin,
            **details,
        }
        self.trade_log.append(entry)
        
        # Append to trade log file
        with open("trades.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def find_opportunities(self) -> list[dict]:
        """Scanne les funding rates et retourne les meilleures opportunités."""
        all_rates = self.client.get_all_funding_rates()
        
        opportunities = []
        for coin, data in all_rates.items():
            # Filtres
            if self.config.allowed_coins and coin not in self.config.allowed_coins:
                continue
            if coin in self.config.blocked_coins:
                continue
            if data["volume24h"] < self.config.min_volume_24h:
                continue
            if data["openInterest"] < self.config.min_open_interest:
                continue
            
            funding_rate = data["funding"]
            abs_funding_pct = abs(funding_rate) * 100
            
            if abs_funding_pct < self.config.min_funding_pct:
                continue
            
            # Vérifier que le funding couvre les fees en un temps raisonnable
            hours_to_breakeven = self.config.roundtrip_fee_pct / (abs(funding_rate) * 100) if abs(funding_rate) > 0 else 999
            
            direction = "SHORT" if funding_rate > 0 else "LONG"
            
            opportunities.append({
                "coin": coin,
                "funding_rate": funding_rate,
                "funding_pct": funding_rate * 100,
                "direction": direction,
                "mark_price": data["markPx"],
                "volume_24h": data["volume24h"],
                "open_interest": data["openInterest"],
                "hours_to_breakeven": hours_to_breakeven,
                "hourly_usd_per_1k": abs(funding_rate) * 1000,
                "sz_decimals": data["szDecimals"],
            })
        
        # Trier par funding absolu décroissant
        opportunities.sort(key=lambda x: abs(x["funding_rate"]), reverse=True)
        return opportunities
    
    def should_enter(self, opp: dict) -> bool:
        """Décide si on doit entrer sur une opportunité."""
        # Déjà une position sur ce coin ?
        if opp["coin"] in self.positions:
            return False
        
        # Nombre max de positions atteint ?
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Funding assez élevé ?
        if abs(opp["funding_pct"]) < self.config.min_funding_pct:
            return False
        
        # Breakeven raisonnable ? (< min_hold_hours)
        if opp["hours_to_breakeven"] > self.config.min_hold_hours:
            log.debug(f"  {opp['coin']}: breakeven {opp['hours_to_breakeven']:.1f}h > {self.config.min_hold_hours}h, skip")
            return False
        
        # Capital disponible ?
        used_capital = sum(
            abs(p.size * p.entry_price) for p in self.positions.values()
        )
        available = self.config.capital - used_capital
        if available < 10:  # Minimum $10 par position
            return False
        
        return True
    
    def enter_position(self, opp: dict):
        """Ouvre une position sur une opportunité."""
        coin = opp["coin"]
        direction = opp["direction"]
        mark_price = opp["mark_price"]
        
        # Calcul de la taille
        used_capital = sum(abs(p.size * p.entry_price) for p in self.positions.values())
        available = min(
            self.config.max_position_usd,
            self.config.capital - used_capital
        )
        
        if available < 10:
            return
        
        # Taille en unités du token
        position_usd = available
        size = position_usd / mark_price
        sz_decimals = opp["sz_decimals"]
        size = round(size, sz_decimals)
        
        if size == 0:
            log.warning(f"  {coin}: taille trop petite, skip")
            return
        
        # Direction: SHORT = sell, LONG = buy
        is_buy = direction == "LONG"
        
        log.info(f"\n{'='*50}")
        log.info(f"🎯 ENTRÉE: {direction} {size} {coin} @ ~${mark_price:.4f}")
        log.info(f"   Funding: {opp['funding_pct']:+.4f}%/h | Breakeven: {opp['hours_to_breakeven']:.1f}h")
        log.info(f"   Position: ~${position_usd:.2f}")
        log.info(f"{'='*50}")
        
        # Placer l'ordre (market pour s'assurer d'être exécuté)
        result = self.client.place_order(coin, is_buy, size, price=None)
        
        # Enregistrer la position
        entry_fee = position_usd * self.config.taker_fee_bps / 10000  # Market order = taker
        
        position = Position(
            coin=coin,
            direction=direction,
            size=size if is_buy else -size,  # Négatif pour short
            entry_price=mark_price,
            entry_time=datetime.now(timezone.utc),
            funding_at_entry=opp["funding_rate"],
            fees_paid=entry_fee,
        )
        
        self.positions[coin] = position
        self.stats.total_positions += 1
        self.stats.total_fees += entry_fee
        
        self._log_trade("ENTER", coin, {
            "direction": direction,
            "size": size,
            "price": mark_price,
            "funding_rate": opp["funding_rate"],
            "position_usd": position_usd,
            "fee": entry_fee,
        })
        
        self._save_state()
    
    def check_exits(self, current_rates: dict):
        """Vérifie si des positions doivent être fermées."""
        to_close = []
        
        for coin, pos in self.positions.items():
            now = datetime.now(timezone.utc)
            hold_hours = (now - pos.entry_time).total_seconds() / 3600
            
            current_price = self.client.get_mid_price(coin)
            if current_price == 0:
                continue
            
            # Calculer le P&L non-réalisé
            if pos.direction == "LONG":
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            else:
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
            
            pos.unrealized_pnl = abs(pos.size) * pos.entry_price * pnl_pct / 100
            
            # Estimer le funding collecté (approximatif, basé sur le funding au moment de l'entrée)
            funding_data = current_rates.get(coin, {})
            current_funding = funding_data.get("funding", 0)
            
            # Le funding est payé toutes les heures; estimer le montant accumulé
            # Note: c'est une approximation, le vrai montant dépend des rates à chaque heure
            estimated_funding_usd = abs(pos.funding_at_entry) * abs(pos.size) * pos.entry_price * hold_hours
            pos.total_funding_collected = estimated_funding_usd
            
            exit_reason = None
            
            # 1. Stop loss
            if pnl_pct < -self.config.stop_loss_pct:
                if hold_hours >= 0.1:  # Au moins 6 minutes pour éviter les faux signaux
                    exit_reason = f"STOP_LOSS (PnL: {pnl_pct:+.2f}%)"
            
            # 2. Take profit
            if pnl_pct > self.config.take_profit_pct:
                exit_reason = f"TAKE_PROFIT (PnL: {pnl_pct:+.2f}%)"
            
            # 3. Funding trop bas (après la période minimum)
            if hold_hours >= self.config.min_hold_hours:
                current_funding_pct = abs(current_funding) * 100
                if current_funding_pct < self.config.exit_funding_pct:
                    exit_reason = f"LOW_FUNDING ({current_funding_pct:.4f}%/h < {self.config.exit_funding_pct}%)"
                
                # Funding a changé de signe → danger
                if pos.direction == "SHORT" and current_funding < 0:
                    exit_reason = f"FUNDING_FLIP (était positif, maintenant négatif)"
                elif pos.direction == "LONG" and current_funding > 0:
                    exit_reason = f"FUNDING_FLIP (était négatif, maintenant positif)"
            
            # 4. Durée max
            if hold_hours >= self.config.max_hold_hours:
                exit_reason = f"MAX_HOLD ({hold_hours:.1f}h)"
            
            if exit_reason:
                to_close.append((coin, exit_reason, current_price, pnl_pct))
            else:
                # Log status
                net = pos.unrealized_pnl + pos.total_funding_collected - pos.fees_paid
                log.info(
                    f"  📊 {coin}: {pos.direction} | {hold_hours:.1f}h | "
                    f"PnL: ${pos.unrealized_pnl:+.4f} | Fund: +${pos.total_funding_collected:.4f} | "
                    f"Fees: -${pos.fees_paid:.4f} | Net: ${net:+.4f}"
                )
        
        # Fermer les positions
        for coin, reason, price, pnl_pct in to_close:
            self.close_position(coin, reason, price)
    
    def close_position(self, coin: str, reason: str, current_price: float):
        """Ferme une position."""
        pos = self.positions.get(coin)
        if not pos:
            return
        
        log.info(f"\n{'='*50}")
        log.info(f"🚪 SORTIE: {coin} | Raison: {reason}")
        
        # Fermer via market order
        result = self.client.close_position(coin, pos.size)
        
        # Calculer le P&L final
        if pos.direction == "LONG":
            price_pnl = (current_price - pos.entry_price) * abs(pos.size)
        else:
            price_pnl = (pos.entry_price - current_price) * abs(pos.size)
        
        exit_fee = abs(pos.size) * current_price * self.config.taker_fee_bps / 10000
        pos.fees_paid += exit_fee
        
        total_pnl = price_pnl + pos.total_funding_collected
        net_pnl = total_pnl - pos.fees_paid
        
        hold_hours = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
        
        log.info(f"   Prix: ${pos.entry_price:.4f} → ${current_price:.4f}")
        log.info(f"   P&L prix: ${price_pnl:+.4f}")
        log.info(f"   Funding collecté: +${pos.total_funding_collected:.4f}")
        log.info(f"   Fees totales: -${pos.fees_paid:.4f}")
        log.info(f"   ═══ Net P&L: ${net_pnl:+.4f} ({hold_hours:.1f}h) ═══")
        log.info(f"{'='*50}")
        
        # Mettre à jour les stats
        pos.status = "CLOSED"
        pos.exit_price = current_price
        pos.exit_time = datetime.now(timezone.utc)
        pos.exit_reason = reason
        
        self.stats.total_pnl += price_pnl
        self.stats.total_funding_collected += pos.total_funding_collected
        self.stats.total_fees += exit_fee
        
        if net_pnl > 0:
            self.stats.winning_positions += 1
        
        self.closed_positions.append(pos)
        del self.positions[coin]
        
        self._log_trade("EXIT", coin, {
            "direction": pos.direction,
            "reason": reason,
            "entry_price": pos.entry_price,
            "exit_price": current_price,
            "price_pnl": price_pnl,
            "funding_collected": pos.total_funding_collected,
            "fees": pos.fees_paid,
            "net_pnl": net_pnl,
            "hold_hours": hold_hours,
        })
        
        self._save_state()
    
    def display_status(self):
        """Affiche le statut actuel."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        mode = "🧪 DRY RUN" if self.config.dry_run else "🔴 LIVE"
        
        print(f"\n┌─ {mode} ─ Funding Farmer ─ {now} {'─'*20}┐")
        
        # Positions ouvertes
        if self.positions:
            print(f"│ 📊 Positions ouvertes: {len(self.positions)}/{self.config.max_positions}")
            for coin, pos in self.positions.items():
                hold_h = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600
                net = pos.unrealized_pnl + pos.total_funding_collected - pos.fees_paid
                print(f"│   {coin}: {pos.direction} {abs(pos.size):.4f} @ ${pos.entry_price:.4f} "
                      f"| {hold_h:.1f}h | net: ${net:+.4f}")
        else:
            print(f"│ 📊 Aucune position ouverte")
        
        # Stats globales
        print(f"│")
        print(f"│ 📈 Stats totales:")
        print(f"│   Positions: {self.stats.total_positions} "
              f"(W: {self.stats.winning_positions} | "
              f"WR: {self.stats.win_rate:.0f}%)")
        print(f"│   Funding collecté: +${self.stats.total_funding_collected:.4f}")
        print(f"│   Fees payées: -${self.stats.total_fees:.4f}")
        print(f"│   Net P&L: ${self.stats.net_pnl:+.4f}")
        print(f"└{'─'*60}┘")
    
    def run(self):
        """Boucle principale du bot."""
        log.info(f"🚀 Démarrage Funding Farmer")
        log.info(f"   Mode: {'DRY RUN' if self.config.dry_run else 'LIVE TRADING'}")
        log.info(f"   Capital: ${self.config.capital}")
        log.info(f"   Max position: ${self.config.max_position_usd:.0f} ({self.config.max_position_pct*100:.0f}%)")
        log.info(f"   Max positions simultanées: {self.config.max_positions}")
        log.info(f"   Seuil entrée: {self.config.min_funding_pct}%/h")
        log.info(f"   Seuil sortie: {self.config.exit_funding_pct}%/h")
        log.info(f"   Stop loss: {self.config.stop_loss_pct}%")
        log.info(f"   Durée min: {self.config.min_hold_hours}h | max: {self.config.max_hold_hours}h")
        
        if self.config.allowed_coins:
            log.info(f"   Coins: {self.config.allowed_coins}")
        
        print()
        
        cycle = 0
        try:
            while True:
                cycle += 1
                log.info(f"─── Cycle {cycle} ───")
                
                # 1. Scanner les opportunités
                opportunities = self.find_opportunities()
                
                if opportunities:
                    top3 = opportunities[:3]
                    for opp in top3:
                        log.info(
                            f"  💰 {opp['coin']}: {opp['direction']} "
                            f"{opp['funding_pct']:+.4f}%/h "
                            f"(BE: {opp['hours_to_breakeven']:.1f}h) "
                            f"${opp['hourly_usd_per_1k']:.4f}/h/$1K"
                        )
                
                # 2. Vérifier les sorties sur les positions existantes
                current_rates = self.client.get_all_funding_rates()
                if self.positions:
                    self.check_exits(current_rates)
                
                # 3. Entrer sur de nouvelles opportunités
                for opp in opportunities:
                    if self.should_enter(opp):
                        self.enter_position(opp)
                
                # 4. Afficher le statut
                self.display_status()
                
                # 5. Attendre
                log.info(f"  ⏳ Prochain cycle dans {self.config.scan_interval}s...")
                time.sleep(self.config.scan_interval)
                
        except KeyboardInterrupt:
            log.info("\n⏹ Arrêt du bot")
            self.display_status()
            self._save_state()
            
            # Proposer de fermer les positions
            if self.positions and not self.config.dry_run:
                print("\n⚠️  Positions ouvertes restantes:")
                for coin, pos in self.positions.items():
                    print(f"  - {coin}: {pos.direction} {abs(pos.size):.4f}")
                resp = input("\nFermer toutes les positions ? (y/n): ")
                if resp.lower() == "y":
                    for coin in list(self.positions.keys()):
                        price = self.client.get_mid_price(coin)
                        self.close_position(coin, "MANUAL_STOP", price)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Funding Rate Farmer — Hyperliquid")
    
    # Mode
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Mode simulation (défaut: activé)")
    parser.add_argument("--live", action="store_true",
                       help="Mode live trading")
    
    # Capital & sizing
    parser.add_argument("--capital", type=float, default=110.0,
                       help="Capital total en USD (défaut: 110)")
    parser.add_argument("--max-pos", type=float, default=0.40,
                       help="Max %% du capital par position (défaut: 0.40)")
    parser.add_argument("--max-positions", type=int, default=3,
                       help="Nombre max de positions (défaut: 3)")
    
    # Stratégie
    parser.add_argument("--threshold", type=float, default=0.015,
                       help="Seuil min funding en %%/h pour entrer (défaut: 0.015)")
    parser.add_argument("--exit-threshold", type=float, default=0.005,
                       help="Seuil funding pour sortir (défaut: 0.005)")
    parser.add_argument("--stop-loss", type=float, default=0.8,
                       help="Stop loss en %% (défaut: 0.8)")
    parser.add_argument("--min-hold", type=float, default=2.0,
                       help="Durée min de holding en heures (défaut: 2)")
    parser.add_argument("--max-hold", type=float, default=48.0,
                       help="Durée max de holding en heures (défaut: 48)")
    
    # Filtres
    parser.add_argument("--coins", type=str, default="",
                       help="Coins autorisés, séparés par virgules (ex: ETH,SOL,BTC)")
    parser.add_argument("--min-volume", type=float, default=100_000,
                       help="Volume 24h minimum en USD (défaut: 100000)")
    
    # Opérationnel
    parser.add_argument("--interval", type=int, default=60,
                       help="Intervalle de scan en secondes (défaut: 60)")
    parser.add_argument("--testnet", action="store_true",
                       help="Utiliser le testnet")
    
    args = parser.parse_args()
    
    # Build config
    config = FarmerConfig(
        secret_key=os.getenv("HL_SECRET_KEY", ""),
        account_address=os.getenv("HL_ACCOUNT_ADDRESS", ""),
        mainnet=not args.testnet,
        min_funding_pct=args.threshold,
        exit_funding_pct=args.exit_threshold,
        capital=args.capital,
        max_position_pct=args.max_pos,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        min_hold_hours=args.min_hold,
        max_hold_hours=args.max_hold,
        min_volume_24h=args.min_volume,
        allowed_coins=[c.strip().upper() for c in args.coins.split(",") if c.strip()] if args.coins else [],
        scan_interval=args.interval,
        dry_run=not args.live,
    )
    
    # Warnings
    if config.dry_run:
        print("╔══════════════════════════════════════════════════╗")
        print("║  🧪 MODE SIMULATION — Aucun ordre ne sera placé ║")
        print("║  Utilise --live pour trader réellement           ║")
        print("╚══════════════════════════════════════════════════╝")
    else:
        print("╔══════════════════════════════════════════════════╗")
        print("║  🔴 MODE LIVE — Les ordres seront exécutés !     ║")
        print("╚══════════════════════════════════════════════════╝")
        
        if not config.secret_key:
            print("❌ HL_SECRET_KEY manquant dans .env")
            sys.exit(1)
        if not config.account_address:
            print("❌ HL_ACCOUNT_ADDRESS manquant dans .env")
            sys.exit(1)
        
        # Confirmation
        confirm = input("\n⚠️  Confirmer le trading live ? (yes/no): ")
        if confirm.lower() != "yes":
            print("Annulé.")
            sys.exit(0)
    
    # Lancer le bot
    farmer = FundingFarmer(config)
    farmer.run()


if __name__ == "__main__":
    main()
