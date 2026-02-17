from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class TraderProfile:
    address: str
    account_value: float = 0.0
    total_pnl: float = 0.0
    pnl_30d: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    avg_position_size: float = 0.0
    num_trades_30d: int = 0
    max_drawdown: float = 1.0
    sharpe_estimate: float = 0.0
    favorite_coins: list[str] = field(default_factory=list)
    is_copiable: bool = False
    score: float = 0.0


@dataclass(slots=True)
class MonitoredTrader:
    address: str
    score: float
    win_rate: float
    pnl_30d: float
    account_value: float
    total_pnl: float


@dataclass(slots=True)
class Position:
    coin: str
    szi: float
    direction: str
    size: float
    entry_price: float
    leverage: float
    position_value: float
    margin_used: float
    unrealized_pnl: float
    mark_price: float


@dataclass(slots=True)
class PositionEvent:
    type: str
    trader_address: str
    coin: str
    direction: str
    timestamp_ms: int
    old_size: float = 0.0
    new_size: float = 0.0
    size: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    leverage: float = 0.0
    position_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: Optional[float] = None
    duration_hours: Optional[float] = None
    source: str = "polling"
