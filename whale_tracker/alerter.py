from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp

from whale_tracker.config import Config
from whale_tracker.db import Database
from whale_tracker.models import MonitoredTrader, PositionEvent


log = logging.getLogger("whale_tracker.alerter")


@dataclass
class PendingGroup:
    trader: MonitoredTrader
    created_ts: float
    events: list[PositionEvent] = field(default_factory=list)


class TelegramAlerter:
    def __init__(
        self,
        db: Database,
        config: Config,
        token: str = "",
        channel_id: str = "",
        dry_run: bool = False,
    ):
        self.db = db
        self.config = config
        self.token = token
        self.channel_id = channel_id
        self.dry_run = dry_run
        self._session: aiohttp.ClientSession | None = None
        self._pending: dict[str, PendingGroup] = {}

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        return self._session

    def queue_events(self, trader: MonitoredTrader, events: list[PositionEvent]) -> None:
        if not events:
            return

        key = trader.address
        pending = self._pending.get(key)

        if pending is None:
            self._pending[key] = PendingGroup(
                trader=trader,
                created_ts=time.time(),
                events=list(events),
            )
            return

        pending.events.extend(events)

    async def flush_due(self, force: bool = False) -> None:
        now = time.time()
        to_send: list[PendingGroup] = []

        for address, pending in list(self._pending.items()):
            age = now - pending.created_ts
            if force or age >= self.config.group_window_seconds:
                to_send.append(pending)
                self._pending.pop(address, None)

        for pending in to_send:
            await self._send_group(pending)

    def _today_midnight_utc_ts(self) -> int:
        now = datetime.now(tz=timezone.utc)
        midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return int(midnight.timestamp())

    def _can_send(self, trader_address: str) -> tuple[bool, str]:
        sent_today = self.db.count_alerts_since(self._today_midnight_utc_ts())
        if sent_today >= self.config.max_alerts_per_day:
            return False, "daily alert cap reached"

        last_alert_ts = self.db.last_alert_for_trader(trader_address)
        if last_alert_ts is not None:
            elapsed = int(time.time()) - last_alert_ts
            if elapsed < self.config.cooldown_same_trader_seconds:
                return False, "trader cooldown active"

        return True, "ok"

    def _fmt_usd(self, value: float) -> str:
        sign = "+" if value > 0 else ""
        if abs(value) >= 1_000_000:
            return f"{sign}${value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{sign}${value/1_000:.1f}K"
        return f"{sign}${value:.0f}"

    def _fmt_pct(self, value: float) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.1f}%"

    def _format_single(self, trader: MonitoredTrader, event: PositionEvent) -> str:
        rank = self.db.get_trader_rank(trader.address)
        score = f"{trader.score:.1f}"
        win_rate_pct = trader.win_rate * 100

        header = {
            "OPEN": "🐋 WHALE OPEN",
            "CLOSE": "🐋 WHALE CLOSE",
            "INCREASE": "🐋 WHALE ADD",
            "DECREASE": "🐋 WHALE TRIM",
        }.get(event.type, "🐋 WHALE UPDATE")

        lines = [f"{header} - {event.coin} {event.direction.upper()}"]
        lines.append("")
        lines.append(f"Trader: #{rank} | Score: {score}/100")
        lines.append(
            f"Win Rate: {win_rate_pct:.1f}% | 30d PnL: {self._fmt_usd(trader.pnl_30d)} | Account: {self._fmt_usd(trader.account_value)}"
        )

        if event.type == "OPEN":
            pct_account = 0.0
            if trader.account_value > 0:
                pct_account = event.position_value / trader.account_value * 100
            lines.append(
                f"Size: {event.size:.4f} {event.coin} ({self._fmt_usd(event.position_value)})"
            )
            lines.append(f"Entry: ${event.entry_price:,.4f} | Leverage: {event.leverage:.1f}x")
            lines.append(f"Exposure: {pct_account:.1f}% of account")

        elif event.type == "CLOSE":
            pnl = event.realized_pnl if event.realized_pnl is not None else 0.0
            outcome = "✅ WIN" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
            lines.append(f"Result: {outcome} | PnL: {self._fmt_usd(pnl)}")
            if event.entry_price > 0 or event.exit_price > 0:
                lines.append(f"Entry: ${event.entry_price:,.4f} -> Exit: ${event.exit_price:,.4f}")

        elif event.type in {"INCREASE", "DECREASE"}:
            delta = event.new_size - event.old_size
            delta_pct = (delta / event.old_size * 100) if event.old_size > 0 else 0.0
            lines.append(
                f"Size: {event.old_size:.4f} -> {event.new_size:.4f} ({self._fmt_pct(delta_pct)})"
            )
            if math.isfinite(event.unrealized_pnl):
                lines.append(f"Unrealized PnL: {self._fmt_usd(event.unrealized_pnl)}")

        lines.append("")
        lines.append(f"https://app.hyperliquid.xyz/explorer/address/{trader.address}")
        return "\n".join(lines)

    def _format_grouped(self, trader: MonitoredTrader, events: list[PositionEvent]) -> str:
        rank = self.db.get_trader_rank(trader.address)

        lines = [f"🐋 WHALE FLOW - {len(events)} updates", ""]
        lines.append(f"Trader: #{rank} | Score: {trader.score:.1f}/100 | Address: {trader.address[:10]}...")
        lines.append("")

        for event in events:
            if event.type == "OPEN":
                lines.append(
                    f"- OPEN {event.coin} {event.direction.upper()} ({self._fmt_usd(event.position_value)})"
                )
            elif event.type == "CLOSE":
                pnl = event.realized_pnl if event.realized_pnl is not None else 0.0
                lines.append(f"- CLOSE {event.coin} {event.direction.upper()} ({self._fmt_usd(pnl)})")
            elif event.type == "INCREASE":
                change = ((event.new_size - event.old_size) / event.old_size * 100) if event.old_size > 0 else 0.0
                lines.append(f"- ADD {event.coin} {event.direction.upper()} ({self._fmt_pct(change)})")
            elif event.type == "DECREASE":
                change = ((event.old_size - event.new_size) / event.old_size * 100) if event.old_size > 0 else 0.0
                lines.append(f"- TRIM {event.coin} {event.direction.upper()} (-{change:.1f}%)")

        lines.append("")
        lines.append(f"https://app.hyperliquid.xyz/explorer/address/{trader.address}")
        return "\n".join(lines)

    async def _send_telegram(self, text: str) -> bool:
        if self.dry_run:
            log.info("[DRY RUN ALERT]\n%s", text)
            return True

        if not self.token or not self.channel_id:
            log.warning("Telegram token/channel missing. Enable --dry-run or pass credentials.")
            return False

        session = await self._ensure_session()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        payload = {
            "chat_id": self.channel_id,
            "text": text,
            "disable_web_page_preview": True,
        }

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    log.warning("Telegram send failed (%s): %s", resp.status, body)
                    return False
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("Telegram send exception: %s", exc)
            return False

    async def _send_group(self, pending: PendingGroup) -> None:
        trader = pending.trader
        events = pending.events
        if not events:
            return

        ok, reason = self._can_send(trader.address)
        if not ok:
            log.info("Skipping alert for %s: %s", trader.address, reason)
            return

        text = self._format_single(trader, events[0]) if len(events) == 1 else self._format_grouped(trader, events)
        sent = await self._send_telegram(text)

        if sent:
            self.db.add_alert_log(
                trader_address=trader.address,
                sent_at=int(time.time()),
                event_count=len(events),
                summary=",".join(e.type for e in events[:8]),
            )
