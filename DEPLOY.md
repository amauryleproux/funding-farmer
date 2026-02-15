# ⚡ HyperPulse — Deployment Guide

## Quick Start (5 minutes)

### 1. Setup on your VPS

```bash
# Créer le dossier
mkdir ~/hyperpulse && cd ~/hyperpulse

# Copier les fichiers
# - hyperpulse_bot.py (ce bot)
# - squeeze_detector.py (ton détecteur existant — inchangé)

# Installer les dépendances
pip install requests pandas numpy python-telegram-bot --break-system-packages
```

### 2. Créer le bot Telegram

1. Ouvrir Telegram → chercher `@BotFather`
2. Envoyer `/newbot`
3. Nom: `HyperPulse Bot`
4. Username: `HyperPulse_xyz_bot` (ou similaire dispo)
5. Copier le **token** (ex: `7123456789:AAH...`)

### 3. Créer le channel Telegram

1. Créer un channel: "HyperPulse Alerts" (public: `@hyperpulse_alerts`)
2. Ajouter le bot comme **administrateur** du channel
3. Donner au bot le droit de **poster des messages**

### 4. Lancer le bot

```bash
# Mode test (terminal seulement, pas de Telegram)
python hyperpulse_bot.py --dry-run

# Mode live
python hyperpulse_bot.py \
    --telegram-token "7123456789:AAH..." \
    --channel-id "@hyperpulse_alerts"

# Avec env vars (recommandé)
export HYPERPULSE_TG_TOKEN="7123456789:AAH..."
export HYPERPULSE_TG_CHANNEL="@hyperpulse_alerts"
python hyperpulse_bot.py
```

### 5. Lancer en background (systemd)

```bash
sudo tee /etc/systemd/system/hyperpulse.service << 'EOF'
[Unit]
Description=HyperPulse Squeeze Alert Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/hyperpulse
Environment=HYPERPULSE_TG_TOKEN=your_token_here
Environment=HYPERPULSE_TG_CHANNEL=@your_channel
ExecStart=/usr/bin/python3 hyperpulse_bot.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable hyperpulse
sudo systemctl start hyperpulse

# Voir les logs
sudo journalctl -u hyperpulse -f
```

## Configuration

| Param | Default | Description |
|-------|---------|-------------|
| `--scan-interval` | 300 | Secondes entre scans (5 min) |
| `--min-score` | 0.55 | Score squeeze minimum |
| `--min-confidence` | 0.60 | Confiance direction minimum |
| `--min-volume` | 100000 | Volume 24h min ($) |
| `--max-volume` | 500000000 | Volume 24h max ($) |
| `--db` | hyperpulse.db | Fichier SQLite |

## Architecture

```
hyperpulse_bot.py          ← Bot principal (standalone)
  ├── HyperliquidData      ← Fetch candles + funding via API
  ├── SqueezeDetector      ← Ton détecteur (importé de squeeze_detector.py)
  ├── SignalTracker         ← Log + résolution des signaux (SQLite)
  ├── TelegramSender       ← Envoi d'alertes
  └── HyperPulseBot        ← Orchestrateur (boucle scan → alert)

squeeze_detector.py        ← Ton code existant (inchangé)
hyperpulse.db              ← Base SQLite (créée automatiquement)
```

## Fichiers requis

Seuls 2 fichiers sont nécessaires :
- `hyperpulse_bot.py` — Le bot
- `squeeze_detector.py` — Ton détecteur existant (aucune modification)

**Pas besoin de** : `squeeze_auto_trader.py`, `squeeze_data_collector.py`, `funding_scanner.py`

## Coexistence avec le trading bot

HyperPulse peut tourner **en parallèle** de ton squeeze_auto_trader :
- Ils utilisent des bases SQLite différentes (`hyperpulse.db` vs `squeeze_data.db`)
- Ils font leurs propres appels API indépendamment
- Aucun conflit possible

## Next Steps

1. **Semaine 1** : Lancer en dry-run, vérifier les signaux
2. **Semaine 2** : Activer Telegram, publier les premiers signaux
3. **Semaine 3** : Créer le compte Twitter, poster les résultats quotidiens
4. **Semaine 4** : Landing page + Stripe → tier payant
