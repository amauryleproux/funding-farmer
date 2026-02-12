#!/usr/bin/env python3
"""
Analyse des trades du Funding Farmer v2.
Deux modes:
  1. --report : Génère un rapport texte à coller dans Claude pour analyse qualitative
  2. --ml     : Feature importance + corrélations pour optimisation
  
Usage:
  python3 analyze_trades.py --report
  python3 analyze_trades.py --ml
  python3 analyze_trades.py --report --ml
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    entries = []
    p = Path(path)
    if not p.exists():
        return []
    for line in p.read_text().strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except:
                pass
    return entries


def pair_trades(trades: list[dict]) -> list[dict]:
    """Associe les ENTER et EXIT pour chaque trade."""
    paired = []
    current_entry = None
    
    for t in trades:
        if t["action"].startswith("ENTER"):
            current_entry = t
        elif t["action"] == "EXIT" and current_entry:
            paired.append({
                "entry": current_entry,
                "exit": t,
                "coin": t["coin"],
                "net": t.get("net", 0),
                "hold_hours": t.get("hold_hours", 0),
                "reason": t.get("reason", ""),
                "mode": t.get("mode", ""),
                "squeeze_triggered": t.get("squeeze_triggered", False),
                "peak_pnl_pct": t.get("peak_pnl_pct", 0),
                # Features d'entrée
                "entry_funding": current_entry.get("funding_rate", 0),
                "entry_squeeze_score": current_entry.get("squeeze_score", 0),
                "entry_premium": current_entry.get("premium_pct", 0),
                "entry_volatility": current_entry.get("volatility_24h", 0),
                "entry_funding_accel": current_entry.get("funding_accel", 0),
                "entry_fv_ratio": current_entry.get("funding_vol_ratio", 0),
                # Market state
                "entry_market": current_entry.get("market_state", {}),
                "exit_market": t.get("market_state_exit", {}),
            })
            current_entry = None
    
    return paired


def generate_report(trades: list[dict], snapshots: list[dict]) -> str:
    """Génère un rapport texte pour analyse par LLM."""
    paired = pair_trades(trades)
    
    if not paired:
        return "Aucun trade complet (entrée + sortie) trouvé."
    
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT FUNDING FARMER v2 — ANALYSE DES TRADES")
    lines.append(f"Période: {paired[0]['entry']['time'][:16]} → {paired[-1]['exit']['time'][:16]}")
    lines.append(f"Nombre de trades: {len(paired)}")
    lines.append("=" * 70)
    
    # Stats globales
    total_net = sum(t["net"] for t in paired)
    winners = [t for t in paired if t["net"] > 0]
    losers = [t for t in paired if t["net"] <= 0]
    squeezes = [t for t in paired if t["squeeze_triggered"]]
    
    lines.append(f"\n📊 RÉSUMÉ GLOBAL:")
    lines.append(f"  P&L total: ${total_net:+.2f}")
    lines.append(f"  Win rate: {len(winners)}/{len(paired)} ({len(winners)/len(paired)*100:.0f}%)")
    lines.append(f"  Gain moyen (winners): ${sum(t['net'] for t in winners)/max(1,len(winners)):+.2f}")
    lines.append(f"  Perte moyenne (losers): ${sum(t['net'] for t in losers)/max(1,len(losers)):+.2f}")
    lines.append(f"  Squeezes capturés: {len(squeezes)}")
    lines.append(f"  Hold moyen: {sum(t['hold_hours'] for t in paired)/len(paired):.1f}h")
    
    # Par raison de sortie
    reasons = {}
    for t in paired:
        r = t["reason"].split(" ")[0] if t["reason"] else "UNKNOWN"
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl": 0, "trades": []}
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t["net"]
        reasons[r]["trades"].append(t)
    
    lines.append(f"\n📈 PAR RAISON DE SORTIE:")
    for reason, data in sorted(reasons.items(), key=lambda x: x[1]["pnl"], reverse=True):
        avg = data["pnl"] / data["count"]
        lines.append(f"  {reason}: {data['count']} trades, total ${data['pnl']:+.2f}, avg ${avg:+.2f}")
    
    # Par token
    tokens = {}
    for t in paired:
        coin = t["coin"]
        if coin not in tokens:
            tokens[coin] = {"count": 0, "pnl": 0}
        tokens[coin]["count"] += 1
        tokens[coin]["pnl"] += t["net"]
    
    lines.append(f"\n🪙 PAR TOKEN:")
    for coin, data in sorted(tokens.items(), key=lambda x: x[1]["pnl"], reverse=True):
        lines.append(f"  {coin}: {data['count']} trades, ${data['pnl']:+.2f}")
    
    # Détail de chaque trade
    lines.append(f"\n{'='*70}")
    lines.append("📋 DÉTAIL DES TRADES:")
    lines.append("=" * 70)
    
    for i, t in enumerate(paired, 1):
        e = t["entry"]
        x = t["exit"]
        sq_tag = " 🚀SQUEEZE" if t["squeeze_triggered"] else ""
        peak_tag = f" (peak: {t['peak_pnl_pct']:+.1f}%)" if t["peak_pnl_pct"] > 0 else ""
        
        lines.append(f"\n  Trade #{i}: {t['coin']} ({t['mode']}){sq_tag}")
        lines.append(f"    Entrée: {e['time'][:19]} @ ${e.get('price', 0):.6f}")
        lines.append(f"    Sortie: {x['time'][:19]} | Raison: {t['reason']}")
        lines.append(f"    Hold: {t['hold_hours']:.1f}h | Net: ${t['net']:+.4f}{peak_tag}")
        
        # Features d'entrée
        lines.append(f"    Features entrée:")
        lines.append(f"      Funding: {t['entry_funding']*100:+.4f}%/h")
        lines.append(f"      Squeeze Score: {t['entry_squeeze_score']}/100")
        lines.append(f"      Premium: {t['entry_premium']:+.2f}%")
        lines.append(f"      Volatilité: {t['entry_volatility']:.1f}%")
        lines.append(f"      Funding accel: {t['entry_funding_accel']:+.0f}%")
        
        # Market state entrée vs sortie
        em = t.get("entry_market", {})
        xm = t.get("exit_market", {})
        if em and xm:
            lines.append(f"    Évolution:")
            lines.append(f"      Funding: {em.get('funding_rate',0)*100:+.4f} → {xm.get('funding_rate',0)*100:+.4f}%/h")
            lines.append(f"      OI: ${em.get('open_interest',0):,.0f} → ${xm.get('open_interest',0):,.0f}")
            lines.append(f"      Premium: {em.get('premium_pct',0):+.2f} → {xm.get('premium_pct',0):+.2f}%")
    
    # Analyse des corrélations (simple)
    if len(paired) >= 5:
        lines.append(f"\n{'='*70}")
        lines.append("🔬 CORRÉLATIONS SIMPLES (feature → profitabilité):")
        lines.append("=" * 70)
        
        features = [
            ("squeeze_score", "entry_squeeze_score"),
            ("funding_rate", "entry_funding"),
            ("premium", "entry_premium"),
            ("volatility", "entry_volatility"),
            ("funding_accel", "entry_funding_accel"),
        ]
        
        for name, key in features:
            # Split au median
            values = [(t.get(key, 0), t["net"]) for t in paired]
            values.sort(key=lambda x: x[0])
            mid = len(values) // 2
            
            low_avg = sum(v[1] for v in values[:mid]) / max(1, mid)
            high_avg = sum(v[1] for v in values[mid:]) / max(1, len(values) - mid)
            
            better = "↑" if high_avg > low_avg else "↓"
            lines.append(f"  {name}: low=${low_avg:+.2f} vs high=${high_avg:+.2f} {better}")
    
    # Questions pour l'IA
    lines.append(f"\n{'='*70}")
    lines.append("❓ QUESTIONS POUR ANALYSE:")
    lines.append("=" * 70)
    lines.append("1. Quels features à l'entrée prédisent les trades gagnants?")
    lines.append("2. Le squeeze score est-il un bon prédicteur de profitabilité?")
    lines.append("3. Quel est le SL optimal basé sur ces données?")
    lines.append("4. Le trailing stop capture-t-il bien les squeezes?")
    lines.append("5. Y a-t-il des tokens systématiquement meilleurs/pires?")
    lines.append("6. La pénalité post-squeeze est-elle calibrée correctement?")
    lines.append("7. Suggestions d'amélioration de la stratégie?")
    
    return "\n".join(lines)


def ml_analysis(trades: list[dict]):
    """Analyse ML basique: feature importance par corrélation."""
    paired = pair_trades(trades)
    
    if len(paired) < 10:
        print(f"⚠️ Seulement {len(paired)} trades — il faut au moins 10 pour une analyse ML fiable.")
        print("  Continue à faire tourner le bot et relance quand tu as plus de data.")
        return
    
    print("=" * 60)
    print("🤖 ANALYSE ML — Feature Importance")
    print("=" * 60)
    
    # Features numériques
    feature_names = [
        "squeeze_score", "funding_rate", "premium", 
        "volatility", "funding_accel", "fv_ratio"
    ]
    feature_keys = [
        "entry_squeeze_score", "entry_funding", "entry_premium",
        "entry_volatility", "entry_funding_accel", "entry_fv_ratio"
    ]
    
    outcomes = [t["net"] for t in paired]
    avg_outcome = sum(outcomes) / len(outcomes)
    
    print(f"\nDataset: {len(paired)} trades, P&L moyen: ${avg_outcome:+.2f}")
    print(f"\nCorrélation feature → P&L (Pearson simplifié):")
    
    for name, key in zip(feature_names, feature_keys):
        values = [float(t.get(key, 0)) for t in paired]
        
        if not values or all(v == values[0] for v in values):
            print(f"  {name:<20} — variance nulle, skip")
            continue
        
        # Pearson correlation
        n = len(values)
        mean_x = sum(values) / n
        mean_y = sum(outcomes) / n
        
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, outcomes)) / n
        std_x = (sum((x - mean_x) ** 2 for x in values) / n) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in outcomes) / n) ** 0.5
        
        if std_x > 0 and std_y > 0:
            corr = cov / (std_x * std_y)
        else:
            corr = 0
        
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"  {name:<20} r={corr:+.3f} {sign}{bar}")
    
    # Top/Bottom trades
    paired.sort(key=lambda t: t["net"], reverse=True)
    
    print(f"\n🏆 TOP 3 TRADES:")
    for t in paired[:3]:
        print(f"  {t['coin']}: ${t['net']:+.2f} | SQ={t['entry_squeeze_score']} "
              f"fund={t['entry_funding']*100:+.3f}% prem={t['entry_premium']:+.2f}% "
              f"vol={t['entry_volatility']:.1f}% | {t['reason']}")
    
    print(f"\n💀 BOTTOM 3 TRADES:")
    for t in paired[-3:]:
        print(f"  {t['coin']}: ${t['net']:+.2f} | SQ={t['entry_squeeze_score']} "
              f"fund={t['entry_funding']*100:+.3f}% prem={t['entry_premium']:+.2f}% "
              f"vol={t['entry_volatility']:.1f}% | {t['reason']}")
    
    # Optimal thresholds
    print(f"\n🎯 SEUILS OPTIMAUX (basé sur win rate):")
    
    for name, key in zip(feature_names, feature_keys):
        values = sorted(set(float(t.get(key, 0)) for t in paired))
        if len(values) < 3:
            continue
        
        best_threshold = None
        best_edge = -999
        
        for threshold in values:
            above = [t for t in paired if float(t.get(key, 0)) >= threshold]
            below = [t for t in paired if float(t.get(key, 0)) < threshold]
            
            if len(above) >= 3 and len(below) >= 3:
                avg_above = sum(t["net"] for t in above) / len(above)
                avg_below = sum(t["net"] for t in below) / len(below)
                edge = avg_above - avg_below
                
                if edge > best_edge:
                    best_edge = edge
                    best_threshold = threshold
        
        if best_threshold is not None:
            print(f"  {name}: seuil optimal = {best_threshold:.4f} (edge: ${best_edge:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="Analyse des trades Funding Farmer")
    parser.add_argument("--report", action="store_true", help="Générer un rapport texte pour LLM")
    parser.add_argument("--ml", action="store_true", help="Analyse ML (feature importance)")
    parser.add_argument("--trades", default="trades_v2.jsonl", help="Fichier trades")
    parser.add_argument("--snapshots", default="snapshots.jsonl", help="Fichier snapshots")
    parser.add_argument("--output", default=None, help="Sauvegarder le rapport dans un fichier")
    args = parser.parse_args()
    
    if not args.report and not args.ml:
        args.report = True
        args.ml = True
    
    trades = load_jsonl(args.trades)
    snapshots = load_jsonl(args.snapshots)
    
    print(f"📂 {len(trades)} entrées dans trades, {len(snapshots)} snapshots")
    
    if args.report:
        report = generate_report(trades, snapshots)
        print(report)
        
        if args.output:
            Path(args.output).write_text(report)
            print(f"\n💾 Rapport sauvegardé: {args.output}")
    
    if args.ml:
        ml_analysis(trades)


if __name__ == "__main__":
    main()
