from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from whale_tracker.alerter import TelegramAlerter
from whale_tracker.client import HyperliquidClient
from whale_tracker.config import Config
from whale_tracker.db import Database
from whale_tracker.models import MonitoredTrader, Position, PositionEvent


log = logging.getLogger("whale_tracker.monitor")


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _fill_time_ms(fill: dict[str, Any]) -> int:
    raw = fill.get("time") or fill.get("timestamp") or fill.get("ts") or 0
    try:
        ts = int(raw)
    except (TypeError, ValueError):
        return 0
    if ts < 10_000_000_000:
        ts *= 1000
    return ts


class WhaleMonitor:
    def __init__(self, db: Database, alerter: TelegramAlerter, config: Config):
        self.db = db
        self.alerter = alerter
        self.config = config
        self.client = HyperliquidClient(config)
        self.position_cache: dict[str, dict[str, Position]] = {}

    async def close(self) -> None:
        await self.client.close()

    async def fetch_positions(self, address: str) -> dict[str, Position]:
        state = await self.client.get_clearinghouse_state(address)
        asset_positions = state.get("assetPositions") if isinstance(state, dict) else []

        if not isinstance(asset_positions, list):
            return {}

        out: dict[str, Position] = {}

        for item in asset_positions:
            if not isinstance(item, dict):
                continue

            raw = item.get("position", item)
            if not isinstance(raw, dict):
                continue

            coin = str(raw.get("coin") or "").upper()
            szi = _to_float(raw.get("szi"), 0.0)
            if not coin or abs(szi) < 1e-12:
                continue

            direction = "long" if szi > 0 else "short"
            size = abs(szi)
            entry_price = _to_float(raw.get("entryPx"), 0.0)

            leverage_raw = raw.get("leverage")
            if isinstance(leverage_raw, dict):
                leverage = _to_float(leverage_raw.get("value"), 0.0)
            else:
                leverage = _to_float(leverage_raw, 0.0)

            mark_price = _to_float(
                raw.get("markPx") or raw.get("markPrice") or raw.get("oraclePx"),
                0.0,
            )
            position_value = _to_float(raw.get("positionValue"), 0.0)
            if position_value <= 0:
                ref_price = mark_price or entry_price
                position_value = size * ref_price

            out[coin] = Position(
                coin=coin,
                szi=szi,
                direction=direction,
                size=size,
                entry_price=entry_price,
                leverage=leverage,
                position_value=position_value,
                margin_used=_to_float(raw.get("marginUsed"), 0.0),
                unrealized_pnl=_to_float(raw.get("unrealizedPnl"), 0.0),
                mark_price=mark_price,
            )

        return out

    async def _enrich_close_event(self, event: PositionEvent) -> None:
        try:
            fills = await self.client.get_user_fills(event.trader_address)
        except Exception:
            return

        recent = sorted(fills, key=_fill_time_ms, reverse=True)
        for fill in recent[:40]:
            coin = str(fill.get("coin") or "").upper()
            if coin != event.coin:
                continue

            direction_text = str(fill.get("dir") or "").lower()
            if "close" not in direction_text:
                continue
            if event.direction not in direction_text:
                continue

            event.exit_price = _to_float(fill.get("px"), event.exit_price)
            event.realized_pnl = _to_float(fill.get("closedPnl"), 0.0)

            fill_time_ms = _fill_time_ms(fill)
            if fill_time_ms:
                duration_h = (event.timestamp_ms - fill_time_ms) / 3_600_000
                if duration_h >= 0:
                    event.duration_hours = duration_h
            return

    def detect_changes(
        self,
        trader: MonitoredTrader,
        previous: dict[str, Position],
        current: dict[str, Position],
        now_ms: int,
    ) -> list[PositionEvent]:
        events: list[PositionEvent] = []

        prev_coins = set(previous.keys())
        curr_coins = set(current.keys())

        for coin in curr_coins - prev_coins:
            pos = current[coin]
            events.append(
                PositionEvent(
                    type="OPEN",
                    trader_address=trader.address,
                    coin=coin,
                    direction=pos.direction,
                    timestamp_ms=now_ms,
                    size=pos.size,
                    new_size=pos.size,
                    entry_price=pos.entry_price,
                    leverage=pos.leverage,
                    position_value=pos.position_value,
                    unrealized_pnl=pos.unrealized_pnl,
                )
            )

        for coin in prev_coins - curr_coins:
            prev_pos = previous[coin]
            events.append(
                PositionEvent(
                    type="CLOSE",
                    trader_address=trader.address,
                    coin=coin,
                    direction=prev_pos.direction,
                    timestamp_ms=now_ms,
                    old_size=prev_pos.size,
                    size=prev_pos.size,
                    position_value=prev_pos.position_value,
                    entry_price=prev_pos.entry_price,
                    leverage=prev_pos.leverage,
                )
            )

        for coin in prev_coins & curr_coins:
            prev_pos = previous[coin]
            curr_pos = current[coin]
            if prev_pos.direction != curr_pos.direction:
                events.append(
                    PositionEvent(
                        type="CLOSE",
                        trader_address=trader.address,
                        coin=coin,
                        direction=prev_pos.direction,
                        timestamp_ms=now_ms,
                        old_size=prev_pos.size,
                        size=prev_pos.size,
                        position_value=prev_pos.position_value,
                        entry_price=prev_pos.entry_price,
                        leverage=prev_pos.leverage,
                    )
                )
                events.append(
                    PositionEvent(
                        type="OPEN",
                        trader_address=trader.address,
                        coin=coin,
                        direction=curr_pos.direction,
                        timestamp_ms=now_ms,
                        size=curr_pos.size,
                        new_size=curr_pos.size,
                        entry_price=curr_pos.entry_price,
                        leverage=curr_pos.leverage,
                        position_value=curr_pos.position_value,
                        unrealized_pnl=curr_pos.unrealized_pnl,
                    )
                )
                continue

            prev_size = abs(prev_pos.szi)
            curr_size = abs(curr_pos.szi)

            if prev_size <= 0:
                continue

            change_ratio = (curr_size - prev_size) / prev_size

            if change_ratio >= self.config.min_size_change_pct:
                events.append(
                    PositionEvent(
                        type="INCREASE",
                        trader_address=trader.address,
                        coin=coin,
                        direction=curr_pos.direction,
                        timestamp_ms=now_ms,
                        old_size=prev_size,
                        new_size=curr_size,
                        size=curr_size,
                        position_value=curr_pos.position_value,
                        entry_price=curr_pos.entry_price,
                        leverage=curr_pos.leverage,
                        unrealized_pnl=curr_pos.unrealized_pnl,
                    )
                )
            elif change_ratio <= -self.config.min_size_change_pct:
                events.append(
                    PositionEvent(
                        type="DECREASE",
                        trader_address=trader.address,
                        coin=coin,
                        direction=curr_pos.direction,
                        timestamp_ms=now_ms,
                        old_size=prev_size,
                        new_size=curr_size,
                        size=curr_size,
                        position_value=curr_pos.position_value,
                        entry_price=curr_pos.entry_price,
                        leverage=curr_pos.leverage,
                        unrealized_pnl=curr_pos.unrealized_pnl,
                    )
                )

        return events

    def _passes_event_filters(self, event: PositionEvent) -> bool:
        if event.type in {"OPEN", "INCREASE"}:
            if event.position_value < self.config.min_position_value_usd:
                return False

        if event.type == "DECREASE" and event.old_size > 0:
            decrease_pct = (event.old_size - event.new_size) / event.old_size
            if decrease_pct < self.config.min_decrease_change_pct:
                return False

        return True

    async def poll_loop(self) -> None:
        while True:
            monitored = self.db.get_monitored_traders(self.config.max_monitored_traders)

            if not monitored:
                log.warning("No monitored traders in DB. Sleeping before retry.")
                await asyncio.sleep(self.config.poll_interval_seconds)
                continue

            for trader in monitored:
                try:
                    now_ms = int(time.time() * 1000)
                    current = await self.fetch_positions(trader.address)
                    previous = self.position_cache.get(trader.address, {})

                    events = self.detect_changes(trader, previous, current, now_ms)
                    filtered_events: list[PositionEvent] = []

                    for event in events:
                        if event.type == "CLOSE":
                            await self._enrich_close_event(event)
                            self.db.close_active_position(
                                trader.address,
                                event.coin,
                                event.timestamp_ms,
                                exit_price=event.exit_price,
                                realized_pnl=event.realized_pnl,
                            )

                        self.db.record_event(event)
                        if self._passes_event_filters(event):
                            filtered_events.append(event)

                    for pos in current.values():
                        self.db.record_position_snapshot(trader.address, pos, now_ms)
                        self.db.upsert_active_position(trader.address, pos, now_ms)

                    if filtered_events:
                        self.alerter.queue_events(trader, filtered_events)

                    self.position_cache[trader.address] = current
                    await asyncio.sleep(self.config.api_delay_seconds)
                except Exception:
                    log.exception("Error polling trader %s", trader.address)

            await self.alerter.flush_due()
            await asyncio.sleep(self.config.poll_interval_seconds)
