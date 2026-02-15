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

## Backtest Squeeze (Hyperliquid)

Pour augmenter la profondeur historique avant backtest:

```bash
# Exemple: backfill Hyperliquid sur 2 ans (si l'API dispose de l'historique)
python squeeze_data_collector.py --collect-all --hl-only --hl-backfill-days 730

# Exemple multi-timeframes (15m, 30m et 1h)
python squeeze_data_collector.py --collect-all --hl-only --hl-backfill-days 365 --hl-intervals 15m,30m,1h

# Update incrémental multi-timeframes
python squeeze_data_collector.py --update --hl-only --hl-intervals 15m,30m,1h
```

Backtest de la stratégie squeeze sur les candles historiques stockées en SQLite:

```bash
# Backtest global (top 50 tokens Hyperliquid par volume)
python3 backtest_hyperliquid.py --db squeeze_data.db

# Fenêtre temporelle précise
python3 backtest_hyperliquid.py --start 2025-01-01 --end 2025-12-31

# Export des résultats
python3 backtest_hyperliquid.py --export-summary bt_summary.csv --export-trades bt_trades.csv
```

Paramètres utiles:
- `--max-tokens` : nombre max de tokens testés
- `--min-candles` : minimum de bougies par token
- `--min-volume` : filtre liquidité 24h
- `--capital-per-token` : capital simulé par token
- `--stop-atr`, `--target-atr`, `--trailing-stop` : risk management

### Backtest Portfolio-Level (fidèle au bot live)

Ce mode simule les contraintes réelles du bot:
- max positions simultanées
- exposition totale max
- cooldowns
- limites journalières (perte max, nb de trades)

```bash
python3 backtest_portfolio_hyperliquid.py \
  --db squeeze_data.db \
  --interval 1h \
  --max-tokens 30 \
  --initial-capital 1000 \
  --export-trades portfolio_trades.csv \
  --export-equity portfolio_equity.csv
```

Exemple setup "pattern reproductible" (issu des diagnostics token):

```bash
python3 backtest_portfolio_hyperliquid.py \
  --db squeeze_data.db \
  --interval 1h \
  --start 2025-07-19 --end 2026-02-12 \
  --max-tokens 25 --min-candles 200 \
  --min-squeeze-score 0.60 \
  --min-direction-confidence 0.50 \
  --min-volume-ratio 0.20 \
  --stop-atr 2.00 --target-atr 2.10 \
  --trailing-stop-pct 0.020 --trailing-activation-pct 0.008 \
  --max-trades-per-day 6 \
  --enable-pattern-filter --pattern-min-rules 3 \
  --pattern-rsi-max 48.0 \
  --pattern-ema-spread-max -0.0030 \
  --pattern-ema-trend-slope-max -0.0018 \
  --pattern-ret8-max -0.0035 \
  --pattern-expected-move-max 0.100
```

Options avancées:
- Split long/short: `--min-*-long`, `--min-*-short`, `--stop-atr-long/short`, `--target-atr-long/short`
- Whitelist dynamique rolling (sans look-ahead): `--enable-dynamic-whitelist --whitelist-lookback-days 30 --whitelist-top-n 12`

### Walk-Forward (robustesse OOS)

Découpe l'historique en fenêtres train/test glissantes pour mesurer la robustesse hors-échantillon:

```bash
python3 backtest_portfolio_hyperliquid.py \
  --db squeeze_data.db \
  --interval 1h \
  --walk-forward \
  --wf-train-days 120 \
  --wf-test-days 30 \
  --wf-step-days 30 \
  --export-wf walk_forward_results.csv
```

Avec optimisation train-only (petite grille, puis test OOS):

```bash
python3 backtest_portfolio_hyperliquid.py \
  --db squeeze_data.db \
  --interval 1h \
  --walk-forward \
  --wf-optimize \
  --wf-train-days 120 \
  --wf-test-days 30 \
  --wf-step-days 30 \
  --wf-max-candidates 40 \
  --export-wf walk_forward_results.csv
```

## Benchmark Multi-Timeframes / Multi-Stratégies

```bash
python3 benchmark_timeframes_strategies.py \
  --db squeeze_data.db \
  --intervals 15m,30m,1h \
  --window-mode both \
  --export strategy_timeframe_benchmark.csv
```

## Presets Prod Par Timeframe

Lance les presets recommandés sur toute la plage disponible de chaque timeframe:

```bash
python3 run_prod_timeframe_presets.py \
  --db squeeze_data.db \
  --export prod_timeframe_results.csv
```

## Diagnostic des patterns par token

Pour comprendre pourquoi certains tokens performent mieux et extraire des patterns plus reproductibles:

```bash
python3 token_pattern_diagnostics.py \
  --db squeeze_data.db \
  --max-tokens 40 \
  --min-token-trades 25 \
  --min-rule-trades 120 \
  --export-token-summary token_summary.csv \
  --export-rules token_rules.csv
```

## Recherche du meilleur setup global

Optimise un setup unique robuste sur plusieurs schemas walk-forward:

```bash
python3 optimize_best_setup.py \
  --db squeeze_data.db \
  --wf-schemes 120:30:30,90:20:20 \
  --max-candidates 60 \
  --workers 4 \
  --top-k 10 \
  --export-results best_setup_search.csv
```

`--workers` permet de paralleliser l'evaluation des candidats (fallback automatique en threads si le process pool n'est pas disponible).

Recherche orientee "boost de rendement" (random search en 2 etapes):

```bash
python3 optimize_profit_boost.py \
  --db squeeze_data.db \
  --start 2025-07-19 --end 2026-02-12 \
  --candidates 40 \
  --stage2-top 8 \
  --export-results profit_boost_results.csv
```
