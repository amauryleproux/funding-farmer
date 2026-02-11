# 💰 Funding Rate Farmer — Hyperliquid

Bot automatisé de farming de funding rates sur Hyperliquid.

## Concept

Sur Hyperliquid, le funding rate est payé **toutes les heures** entre longs et shorts :
- Funding **positif** → les longs paient les shorts → on **SHORT** pour collecter
- Funding **négatif** → les shorts paient les longs → on **LONG** pour collecter

Ce bot scanne les funding rates, identifie les tokens avec des rates élevés, et ouvre des positions pour capturer ce revenu.

## ⚠️ Risques (mode sans hedge)

- **Exposition directionnelle** : le prix peut bouger contre la position
- Le stop-loss protège mais réduit les gains
- Les fees d'entrée/sortie doivent être couvertes par le funding
- Ce n'est PAS du delta-neutral : c'est un pari que le funding > mouvement de prix adverse

## Setup

```bash
pip install -r requirements.txt
```

Fichier `.env` :
```
HL_SECRET_KEY=0x...votre_cle_privee_api...
HL_ACCOUNT_ADDRESS=0x...votre_adresse_wallet...
```

## Usage

### 1. Scanner (monitoring sans trading)

```bash
# Scan one-shot — voir les opportunities actuelles
python funding_scanner.py

# Mode watch — refresh toutes les 60s
python funding_scanner.py --watch

# Filtre plus strict (seulement les fundings > 0.03%/h)
python funding_scanner.py --threshold 0.03

# Historique funding d'un token spécifique
python funding_scanner.py --history ETH
python funding_scanner.py --history DOGE --hours 48

# Avec ton capital réel
python funding_scanner.py --capital 110
```

### 2. Bot Farmer (trading automatisé)

```bash
# Mode simulation (par défaut) — PAS d'ordres réels
python funding_farmer.py --dry-run

# Simulation avec params custom
python funding_farmer.py --capital 110 --threshold 0.02 --stop-loss 0.5

# Limiter à certains tokens
python funding_farmer.py --coins ETH,SOL,BTC

# MODE LIVE (⚠️ ordres réels !)
python funding_farmer.py --live --capital 110
```

### Paramètres du farmer

| Param | Défaut | Description |
|-------|--------|-------------|
| `--threshold` | 0.015 | Funding min pour entrer (%/h) |
| `--exit-threshold` | 0.005 | Funding min pour rester (%/h) |
| `--capital` | 110 | Capital total ($) |
| `--max-pos` | 0.40 | Max % capital par position |
| `--max-positions` | 3 | Positions simultanées max |
| `--stop-loss` | 0.8 | Stop loss (%) |
| `--min-hold` | 2 | Durée min (heures) |
| `--max-hold` | 48 | Durée max (heures) |
| `--interval` | 60 | Fréquence scan (secondes) |

## Économie avec $110

Avec les fees Hyperliquid (maker: 1.5 bps, taker: 4.5 bps) :
- **Coût round-trip** (market orders) : ~$0.10 pour $110 de position
- **Funding à 0.02%/h** : $0.022/h → breakeven en ~4.5h
- **Funding à 0.05%/h** : $0.055/h → breakeven en ~1.8h
- **Potentiel journalier** (funding moyen 0.03%/h) : ~$0.79/jour = ~$24/mois

## Plan de test (2 jours)

1. **Jour 1 matin** : Lance le scanner en mode watch, observe quels tokens ont des fundings élevés
2. **Jour 1 après-midi** : Lance le farmer en dry-run, vérifie que la logique de sélection est bonne
3. **Jour 1 soir** : Si les résultats dry-run sont satisfaisants, passe en live avec stop-loss serré
4. **Jour 2** : Analyse les trades, ajuste les seuils si besoin

## Fichiers générés

- `farmer_state.json` : État du bot (positions, stats)
- `trades.jsonl` : Log de tous les trades
- `funding_scan_*.json` : Exports des scans (avec --json)
