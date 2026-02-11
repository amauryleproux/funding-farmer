#!/usr/bin/env python3
"""
Funding Rate Scanner pour Hyperliquid
======================================
Scanne tous les tokens et affiche les opportunités de funding rate farming.

Le funding sur Hyperliquid est payé toutes les heures.
- Funding positif = les longs paient les shorts → on SHORT pour collecter
- Funding négatif = les shorts paient les longs → on LONG pour collecter

Usage:
    python funding_scanner.py                  # Scan one-shot
    python funding_scanner.py --watch          # Refresh toutes les 60s
    python funding_scanner.py --threshold 0.01 # Filtre custom (0.01% = 1 bps/h)
    python funding_scanner.py --history ETH    # Historique funding d'un token
"""

import requests
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

# ============================================================================
# CONFIG
# ============================================================================

API_URL = "https://api.hyperliquid.xyz/info"

# Seuils par défaut
DEFAULT_THRESHOLD_PCT = 0.005  # 0.005% par heure = 0.5 bps minimum pour afficher
HIGH_THRESHOLD_PCT = 0.02     # 0.02% = 2 bps/h = considéré "élevé"
EXTREME_THRESHOLD_PCT = 0.05  # 0.05% = 5 bps/h = considéré "extrême"


@dataclass
class FundingOpp:
    coin: str
    funding_rate: float         # Taux horaire en décimal (0.0001 = 0.01%)
    funding_pct: float          # En pourcentage (0.01 = 0.01%)
    annualized_pct: float       # Annualisé
    mark_price: float
    open_interest: float        # En USD
    volume_24h: float           # En USD
    direction: str              # "SHORT" ou "LONG" (direction à prendre pour collecter)
    premium: float              # Premium mark vs oracle en %
    hourly_usd_per_1000: float  # $ gagnés par heure pour $1000 de position


def fetch_meta_and_contexts() -> tuple[list[dict], list[dict]]:
    """Récupère metadata + contextes de tous les assets."""
    resp = requests.post(API_URL, json={"type": "metaAndAssetCtxs"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # data = [meta_dict, [asset_ctx_list]]
    meta = data[0]["universe"]
    ctxs = data[1]
    return meta, ctxs


def fetch_funding_history(coin: str, hours: int = 72) -> list[dict]:
    """Récupère l'historique des funding rates d'un token."""
    start_time = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
    resp = requests.post(API_URL, json={
        "type": "fundingHistory",
        "coin": coin,
        "startTime": start_time
    }, timeout=10)
    resp.raise_for_status()
    return resp.json()


def scan_funding_rates(threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> list[FundingOpp]:
    """Scanne tous les tokens et retourne les opportunités triées."""
    meta, ctxs = fetch_meta_and_contexts()
    
    opportunities = []
    
    for asset_meta, ctx in zip(meta, ctxs):
        coin = asset_meta["name"]
        
        try:
            funding_rate = float(ctx.get("funding", "0"))
            mark_price = float(ctx.get("markPx", "0"))
            oracle_price = float(ctx.get("oraclePx", "0"))
            open_interest = float(ctx.get("openInterest", "0"))
            volume_24h = float(ctx.get("dayNtlVlm", "0"))
            
            if mark_price == 0:
                continue
                
            funding_pct = funding_rate * 100  # Convertir en %
            abs_funding_pct = abs(funding_pct)
            
            # Filtre: ne garder que les fundings au-dessus du seuil
            if abs_funding_pct < threshold_pct:
                continue
            
            # Direction à prendre pour collecter le funding
            # Funding positif → les longs paient → on short
            # Funding négatif → les shorts paient → on long
            direction = "SHORT" if funding_rate > 0 else "LONG"
            
            # Premium mark vs oracle
            premium = ((mark_price - oracle_price) / oracle_price * 100) if oracle_price > 0 else 0
            
            # Annualisé (24h * 365j)
            annualized = abs_funding_pct * 24 * 365
            
            # $ par heure pour $1000 de position
            hourly_usd = abs(funding_rate) * 1000
            
            # OI en USD
            oi_usd = open_interest * mark_price
            
            opportunities.append(FundingOpp(
                coin=coin,
                funding_rate=funding_rate,
                funding_pct=funding_pct,
                annualized_pct=annualized,
                mark_price=mark_price,
                open_interest=oi_usd,
                volume_24h=volume_24h,
                direction=direction,
                premium=premium,
                hourly_usd_per_1000=hourly_usd,
            ))
        except (ValueError, KeyError):
            continue
    
    # Trier par funding absolu décroissant
    opportunities.sort(key=lambda x: abs(x.funding_rate), reverse=True)
    return opportunities


def format_usd(value: float) -> str:
    """Formate un montant USD."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.0f}"


def display_opportunities(opps: list[FundingOpp], capital: float = 110.0):
    """Affiche les opportunités dans un tableau formaté."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    print(f"\n{'='*90}")
    print(f"  💰 FUNDING RATE SCANNER — Hyperliquid    |    {now}")
    print(f"  Capital: ${capital:.0f}  |  Maker fee: 1.5 bps  |  Breakeven round-trip: 3 bps")
    print(f"{'='*90}")
    
    if not opps:
        print("\n  ❌ Aucune opportunité au-dessus du seuil.\n")
        return
    
    # Header
    print(f"\n  {'Coin':<10} {'Dir':>5} {'Fund/h':>10} {'Annual':>10} "
          f"{'$/h ($1K)':>10} {'$/h (you)':>10} {'Premium':>9} "
          f"{'OI':>10} {'Vol24h':>10} {'Grade':>6}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*10} {'-'*9} "
          f"{'-'*10} {'-'*10} {'-'*6}")
    
    for opp in opps[:30]:  # Top 30
        abs_fund = abs(opp.funding_pct)
        
        # Grade basé sur le funding
        if abs_fund >= EXTREME_THRESHOLD_PCT:
            grade = "🔥 A+"
        elif abs_fund >= HIGH_THRESHOLD_PCT:
            grade = "⭐ A"
        elif abs_fund >= 0.01:
            grade = "✅ B"
        else:
            grade = "   C"
        
        # Revenu estimé avec le capital de l'utilisateur (position = capital avec levier 1x)
        hourly_user = abs(opp.funding_rate) * capital
        
        # Direction colorée
        dir_str = opp.direction
        
        # Premium warning si > 0.5%
        premium_str = f"{opp.premium:+.3f}%"
        
        print(f"  {opp.coin:<10} {dir_str:>5} {opp.funding_pct:>+.4f}% "
              f"{opp.annualized_pct:>8.1f}% "
              f"${opp.hourly_usd_per_1000:>8.4f} "
              f"${hourly_user:>8.4f} "
              f"{premium_str:>9} "
              f"{format_usd(opp.open_interest):>10} "
              f"{format_usd(opp.volume_24h):>10} "
              f"{grade:>6}")
    
    # Résumé
    print(f"\n  📊 {len(opps)} tokens avec funding élevé")
    
    if opps:
        top = opps[0]
        daily_est = abs(top.funding_rate) * capital * 24
        monthly_est = daily_est * 30
        print(f"\n  🏆 Meilleure opportunité: {top.coin}")
        print(f"     → {top.direction} pour collecter {top.funding_pct:+.4f}%/h = {top.annualized_pct:.0f}% annualisé")
        print(f"     → Avec ${capital:.0f}: ~${daily_est:.2f}/jour, ~${monthly_est:.1f}/mois (si constant)")
        print(f"     → ⚠️  SANS hedge = exposition directionnelle !")
    
    # Calcul du coût d'entrée/sortie
    roundtrip_bps = 3.0  # 1.5 bps maker × 2
    for opp in opps[:5]:
        hours_to_breakeven = roundtrip_bps / (abs(opp.funding_pct) * 100) if abs(opp.funding_pct) > 0 else float('inf')
        if hours_to_breakeven < 100:
            print(f"     {opp.coin}: breakeven fees en {hours_to_breakeven:.1f}h")
    
    print()


def display_funding_history(coin: str, hours: int = 72):
    """Affiche l'historique des funding rates."""
    print(f"\n📈 Historique funding {coin} (dernières {hours}h)")
    print(f"{'='*60}")
    
    history = fetch_funding_history(coin, hours)
    
    if not history:
        print("  Pas de données.\n")
        return
    
    total_funding = 0.0
    positive_count = 0
    negative_count = 0
    max_rate = 0.0
    min_rate = 0.0
    
    print(f"\n  {'Time (UTC)':<20} {'Funding Rate':>14} {'Direction':>10} {'$/1K':>10}")
    print(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*10}")
    
    for entry in history[-48:]:  # Dernières 48h max
        ts = datetime.fromtimestamp(entry["time"] / 1000, tz=timezone.utc)
        rate = float(entry["fundingRate"])
        rate_pct = rate * 100
        total_funding += rate
        
        if rate > 0:
            positive_count += 1
            direction = "SHORT ✓"
        else:
            negative_count += 1
            direction = "LONG ✓"
        
        max_rate = max(max_rate, rate)
        min_rate = min(min_rate, rate)
        
        usd_per_1k = abs(rate) * 1000
        
        print(f"  {ts.strftime('%m-%d %H:%M'):>20} {rate_pct:>+.5f}% {direction:>10} ${usd_per_1k:>.4f}")
    
    # Stats
    n = len(history)
    avg_rate = (total_funding / n) if n > 0 else 0
    avg_pct = avg_rate * 100
    
    print(f"\n  📊 Stats sur {n} périodes ({n}h):")
    print(f"     Funding moyen: {avg_pct:+.5f}%/h")
    print(f"     Funding cumulé: {total_funding * 100:+.4f}%")
    print(f"     Max: {max_rate * 100:+.5f}% | Min: {min_rate * 100:+.5f}%")
    print(f"     Positif: {positive_count} fois ({positive_count/n*100:.0f}%) | Négatif: {negative_count} fois ({negative_count/n*100:.0f}%)")
    
    # Direction dominante
    if positive_count > negative_count * 1.5:
        print(f"     → Direction dominante: SHORT (funding majoritairement positif)")
    elif negative_count > positive_count * 1.5:
        print(f"     → Direction dominante: LONG (funding majoritairement négatif)")
    else:
        print(f"     → Funding mixte, pas de direction dominante claire")
    
    # Estimation revenus
    daily_avg_usd_per_1k = abs(avg_rate) * 1000 * 24
    print(f"     → Revenu estimé moyen: ${daily_avg_usd_per_1k:.2f}/jour pour $1K de position")
    print()


def main():
    parser = argparse.ArgumentParser(description="Funding Rate Scanner - Hyperliquid")
    parser.add_argument("--watch", action="store_true", help="Mode continu (refresh toutes les 60s)")
    parser.add_argument("--interval", type=int, default=60, help="Intervalle de refresh en secondes (défaut: 60)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT, 
                       help=f"Seuil minimum de funding en %% (défaut: {DEFAULT_THRESHOLD_PCT})")
    parser.add_argument("--capital", type=float, default=110.0, help="Capital disponible en USD (défaut: 110)")
    parser.add_argument("--history", type=str, help="Afficher l'historique funding d'un token (ex: ETH)")
    parser.add_argument("--hours", type=int, default=72, help="Heures d'historique (défaut: 72)")
    parser.add_argument("--json", action="store_true", help="Exporter en JSON")
    
    args = parser.parse_args()
    
    # Mode historique
    if args.history:
        display_funding_history(args.history.upper(), args.hours)
        return
    
    if args.watch:
        print("🔄 Mode watch activé (Ctrl+C pour arrêter)")
        try:
            while True:
                opps = scan_funding_rates(args.threshold)
                
                # Clear screen
                print("\033[2J\033[H", end="")
                
                display_opportunities(opps, args.capital)
                
                if args.json:
                    export = [{
                        "coin": o.coin,
                        "funding_pct_hourly": o.funding_pct,
                        "annualized_pct": o.annualized_pct,
                        "direction": o.direction,
                        "mark_price": o.mark_price,
                        "open_interest_usd": o.open_interest,
                        "volume_24h_usd": o.volume_24h,
                    } for o in opps]
                    filename = f"funding_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, "w") as f:
                        json.dump(export, f, indent=2)
                    print(f"  💾 Exporté: {filename}")
                
                print(f"  ⏳ Prochain refresh dans {args.interval}s...")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n  ⏹ Arrêt du scanner.")
    else:
        opps = scan_funding_rates(args.threshold)
        display_opportunities(opps, args.capital)
        
        if args.json:
            export = [{
                "coin": o.coin,
                "funding_pct_hourly": o.funding_pct,
                "annualized_pct": o.annualized_pct,
                "direction": o.direction,
                "mark_price": o.mark_price,
                "open_interest_usd": o.open_interest,
                "volume_24h_usd": o.volume_24h,
            } for o in opps]
            filename = f"funding_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(export, f, indent=2)
            print(f"  💾 Exporté: {filename}")


if __name__ == "__main__":
    main()
