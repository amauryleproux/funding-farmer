from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Telegram
    telegram_token: str = ""
    channel_id: str = ""

    # Database
    db_path: str = "whale_tracker.db"

    # Monitoring
    max_monitored_traders: int = 25
    poll_interval_seconds: float = 10.0
    api_delay_seconds: float = 0.1

    # Alert rules
    max_alerts_per_day: int = 20
    min_position_value_usd: float = 10_000
    min_size_change_pct: float = 0.20
    min_decrease_change_pct: float = 0.30
    cooldown_same_trader_seconds: int = 300
    group_window_seconds: int = 60

    # Scanner criteria
    min_account_value: float = 50_000
    min_win_rate: float = 0.50
    min_pnl_30d: float = 0
    min_trades_30d: int = 10
    max_trades_30d: int = 5000
    min_avg_duration_hours: float = 0.1
    max_avg_duration_hours: float = 168.0

    # Scanner schedule
    refresh_interval_hours: int = 4

    # API
    api_base_url: str = "https://api.hyperliquid.xyz"
    api_timeout_seconds: int = 20
    api_retries: int = 3

    # Copy-trading sizing
    my_account_size: float = 200.0       # User's account in USD
    max_exposure_pct: float = 0.50       # Max 50% of account per position
    max_positions: int = 2               # Max simultaneous positions to suggest
    min_position_usd: float = 10.0       # Hyperliquid minimum

    # Fallback list if leaderboard endpoint fails.
    # You can also set HL_SEED_TRADERS=0x...,0x...
    fallback_addresses: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Config":
        env_fallback = os.getenv("HL_SEED_TRADERS", "").strip()
        fallback_addresses = [
            addr.strip() for addr in env_fallback.split(",") if addr.strip()
        ]

        return cls(
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            channel_id=os.getenv("TELEGRAM_CHANNEL_ID", ""),
            db_path=os.getenv("WHALE_DB_PATH", "whale_tracker.db"),
            fallback_addresses=fallback_addresses,
        )
