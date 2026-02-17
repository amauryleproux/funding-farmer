from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from whale_tracker.client import HyperliquidClient
from whale_tracker.config import Config
from whale_tracker.db import Database
from whale_tracker.models import TraderProfile


log = logging.getLogger("whale_tracker.scanner")

# ── Vault addresses for trader discovery ──────────────────────────────
# HLP Main vault — largest liquidity pool on Hyperliquid
VAULT_ADDRESSES: list[str] = [
    "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",
]

# Known mega-whale seed addresses (always included in scans)
SEED_ADDRESSES: list[str] = [
    "0x31ca8395cf837de08b24da3f660e77761dfb974b",
]


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


def _parse_fill_time_ms(fill: dict[str, Any]) -> int:
    candidates = [fill.get("time"), fill.get("timestamp"), fill.get("ts")]
    for candidate in candidates:
        try:
            ts = int(candidate)
            # Convert seconds to ms if needed.
            if ts < 10_000_000_000:
                ts *= 1000
            return ts
        except (TypeError, ValueError):
            continue
    return 0


def _extract_account_value(state: dict[str, Any]) -> float:
    if not state:
        return 0.0

    margin_summary = state.get("marginSummary") or {}
    if isinstance(margin_summary, dict):
        for key in ("accountValue", "account_value", "equity"):
            if key in margin_summary:
                return _to_float(margin_summary.get(key), 0.0)

    cross_summary = state.get("crossMarginSummary") or {}
    if isinstance(cross_summary, dict):
        for key in ("accountValue", "account_value", "equity"):
            if key in cross_summary:
                return _to_float(cross_summary.get(key), 0.0)

    for key in ("accountValue", "account_value", "equity"):
        if key in state:
            return _to_float(state.get(key), 0.0)

    return 0.0


def _estimate_avg_duration_hours(fills_sorted: list[dict[str, Any]]) -> float:
    if not fills_sorted:
        return 0.0

    active_open_ts: dict[tuple[str, str], int] = {}
    durations_hours: list[float] = []

    for fill in fills_sorted:
        direction_text = str(fill.get("dir") or "").lower()
        coin = str(fill.get("coin") or "").upper()
        ts_ms = _parse_fill_time_ms(fill)
        if not coin or not ts_ms:
            continue

        side = "long" if "long" in direction_text else "short" if "short" in direction_text else ""
        if not side:
            side_value = str(fill.get("side") or "").lower()
            side = "long" if side_value in {"b", "buy"} else "short" if side_value in {"a", "sell"} else ""
        if not side:
            continue

        key = (coin, side)
        is_open = "open" in direction_text
        is_close = "close" in direction_text

        if is_open:
            active_open_ts[key] = ts_ms
        elif is_close and key in active_open_ts:
            duration_h = (ts_ms - active_open_ts[key]) / 3_600_000
            if duration_h > 0:
                durations_hours.append(duration_h)
            active_open_ts.pop(key, None)

    if durations_hours:
        return float(sum(durations_hours) / len(durations_hours))

    # Fallback proxy: average spacing between fills, divided by 2 as rough hold-time proxy.
    timestamps = [_parse_fill_time_ms(f) for f in fills_sorted if _parse_fill_time_ms(f) > 0]
    if len(timestamps) < 2:
        return 0.0
    timestamps.sort()
    deltas = [(b - a) / 3_600_000 for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not deltas:
        return 0.0
    return max(0.0, float(sum(deltas) / len(deltas) / 2))


def _estimate_max_drawdown(fills_sorted: list[dict[str, Any]]) -> float:
    pnls: list[float] = []
    for fill in fills_sorted:
        pnl = _to_float(fill.get("closedPnl"), 0.0)
        pnls.append(pnl)

    if not pnls:
        return 1.0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        denominator = max(1.0, abs(peak))
        drawdown = (peak - equity) / denominator
        max_drawdown = max(max_drawdown, drawdown)

    return float(max_drawdown)


def _estimate_sharpe_30d(fills_30d: list[dict[str, Any]]) -> float:
    if not fills_30d:
        return 0.0

    daily_pnl: dict[str, float] = defaultdict(float)
    for fill in fills_30d:
        ts_ms = _parse_fill_time_ms(fill)
        if ts_ms <= 0:
            continue
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        daily_pnl[day] += _to_float(fill.get("closedPnl"), 0.0)

    values = list(daily_pnl.values())
    if len(values) < 2:
        return 0.0

    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev <= 1e-9:
        return 0.0
    # Daily Sharpe annualized using sqrt(365) for 24/7 market.
    return float((mean / stdev) * math.sqrt(365.0))


class TraderScanner:
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config
        self.client = HyperliquidClient(config)

    async def close(self) -> None:
        await self.client.close()

    async def discover_addresses_from_vaults(self) -> list[str]:
        """Discover trader addresses from Hyperliquid vault depositors."""
        addresses: list[str] = []

        for vault_addr in VAULT_ADDRESSES:
            try:
                details = await self.client.get_vault_details(vault_addr)
                if not details:
                    log.warning("Vault %s returned empty details", vault_addr[:10])
                    continue

                # Extract follower/depositor addresses
                followers = details.get("followers") or []
                for follower in followers:
                    user = follower.get("user", "")
                    if user:
                        addresses.append(user)

                # Also include the vault leader
                leader = details.get("leader", "")
                if leader:
                    addresses.append(leader)

                log.info(
                    "Vault %s: %d followers + leader discovered",
                    vault_addr[:10],
                    len(followers),
                )
                await asyncio.sleep(self.config.api_delay_seconds)
            except Exception as exc:
                log.warning("Vault %s fetch failed: %s", vault_addr[:10], exc)

        return addresses

    async def discover_traders(self) -> list[str]:
        """Combine vault-based discovery, seed addresses, and config fallbacks."""
        addresses: list[str] = []

        # 1. Vault-based discovery (primary source)
        vault_addresses = await self.discover_addresses_from_vaults()
        addresses.extend(vault_addresses)
        if vault_addresses:
            log.info("Discovered %d addresses from vaults", len(vault_addresses))

        # 2. Hardcoded seed addresses (known mega-whales)
        addresses.extend(SEED_ADDRESSES)

        # 3. Config fallback addresses (from env HL_SEED_TRADERS)
        if self.config.fallback_addresses:
            addresses.extend(self.config.fallback_addresses)

        # Deduplicate
        deduped = []
        seen: set[str] = set()
        for addr in addresses:
            norm = addr.lower()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(norm)

        log.info("Total unique addresses to analyze: %d", len(deduped))
        return deduped

    async def analyze_trader(self, address: str) -> TraderProfile:
        state = await self.client.get_clearinghouse_state(address)
        fills = await self.client.get_user_fills(address)

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - 30 * 24 * 3600 * 1000

        fills_sorted = sorted(fills, key=_parse_fill_time_ms)
        fills_30d = [f for f in fills_sorted if _parse_fill_time_ms(f) >= cutoff_ms]

        account_value = _extract_account_value(state)

        closed_pnls = [_to_float(f.get("closedPnl"), 0.0) for f in fills_sorted]
        total_pnl = float(sum(closed_pnls))

        closed_pnls_30d = [_to_float(f.get("closedPnl"), 0.0) for f in fills_30d]
        pnl_30d = float(sum(closed_pnls_30d))

        realized_trade_pnls = [p for p in closed_pnls if abs(p) > 1e-9]
        wins = sum(1 for p in realized_trade_pnls if p > 0)
        win_rate = float(wins / len(realized_trade_pnls)) if realized_trade_pnls else 0.0

        notionals = []
        for fill in fills_30d:
            px = _to_float(fill.get("px"), 0.0)
            sz = abs(_to_float(fill.get("sz"), 0.0))
            notional = px * sz
            if notional > 0:
                notionals.append(notional)
        avg_position_size = float(sum(notionals) / len(notionals)) if notionals else 0.0

        num_trades_30d = len(fills_30d)
        avg_trade_duration = _estimate_avg_duration_hours(fills_sorted)
        max_drawdown = _estimate_max_drawdown(fills_sorted)
        sharpe_estimate = _estimate_sharpe_30d(fills_30d)

        coins = [str(fill.get("coin") or "").upper() for fill in fills_30d if fill.get("coin")]
        favorite_coins = [coin for coin, _ in Counter(coins).most_common(5)]

        consistency_bonus = max(0.0, 1.0 - min(max_drawdown, 1.0))
        score = (
            win_rate * 25
            + min(pnl_30d / 100_000, 1.0) * 25
            + min(sharpe_estimate / 3.0, 1.0) * 25
            + consistency_bonus * 25
        )
        score = max(0.0, min(100.0, score))

        is_copiable = all(
            [
                account_value >= self.config.min_account_value,
                pnl_30d > self.config.min_pnl_30d,
                win_rate >= self.config.min_win_rate,
                avg_trade_duration >= self.config.min_avg_duration_hours,
                avg_trade_duration <= self.config.max_avg_duration_hours,
                num_trades_30d >= self.config.min_trades_30d,
                num_trades_30d <= self.config.max_trades_30d,
            ]
        )

        return TraderProfile(
            address=address.lower(),
            account_value=account_value,
            total_pnl=total_pnl,
            pnl_30d=pnl_30d,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            avg_position_size=avg_position_size,
            num_trades_30d=num_trades_30d,
            max_drawdown=max_drawdown,
            sharpe_estimate=sharpe_estimate,
            favorite_coins=favorite_coins,
            is_copiable=is_copiable,
            score=score,
        )

    async def _has_active_positions(self, address: str) -> bool:
        """Check if a trader has at least one open position."""
        try:
            state = await self.client.get_clearinghouse_state(address)
            positions = state.get("assetPositions") or []
            for pos in positions:
                item = pos.get("position") or pos
                szi = _to_float(item.get("szi"), 0.0)
                if abs(szi) > 0:
                    return True
            return False
        except Exception:
            return False

    async def full_scan(self) -> list[TraderProfile]:
        addresses = await self.discover_traders()
        if not addresses:
            log.warning("No trader addresses found. Add HL_SEED_TRADERS as fallback.")
            return []

        # Pre-filter: only analyze traders with active positions and sufficient equity
        filtered_addresses: list[str] = []
        for idx, address in enumerate(addresses, start=1):
            try:
                state = await self.client.get_clearinghouse_state(address)
                account_value = _extract_account_value(state)

                # Filter: account_value >= $50K
                if account_value < self.config.min_account_value:
                    continue

                # Filter: at least 1 active position
                positions = state.get("assetPositions") or []
                has_position = any(
                    abs(_to_float((p.get("position") or p).get("szi"), 0.0)) > 0
                    for p in positions
                )
                if not has_position:
                    continue

                filtered_addresses.append(address)
            except Exception as exc:
                log.warning("Pre-filter failed for %s: %s", address[:10], exc)

            if idx % 20 == 0:
                log.info("Pre-filter progress: %d/%d checked, %d passed", idx, len(addresses), len(filtered_addresses))
            await asyncio.sleep(self.config.api_delay_seconds)

        log.info(
            "Pre-filter complete: %d/%d addresses have active positions and >= $%.0fK equity",
            len(filtered_addresses),
            len(addresses),
            self.config.min_account_value / 1000,
        )

        # Full analysis on filtered addresses only
        profiles: list[TraderProfile] = []
        for idx, address in enumerate(filtered_addresses, start=1):
            try:
                profile = await self.analyze_trader(address)
                self.db.upsert_trader(profile)
                profiles.append(profile)
            except Exception as exc:
                log.warning("Failed to analyze trader %s (%s)", address[:10], exc)

            if idx % 10 == 0:
                log.info("Scanner progress: %d/%d", idx, len(filtered_addresses))
            await asyncio.sleep(self.config.api_delay_seconds)

        copiable = [p for p in profiles if p.is_copiable]
        selected = sorted(copiable, key=lambda p: p.score, reverse=True)[: self.config.max_monitored_traders]
        self.db.set_monitored_bulk([p.address for p in selected])

        log.info(
            "Scan complete: %d analyzed, %d copiable, %d monitored",
            len(profiles),
            len(copiable),
            len(selected),
        )
        return profiles

    async def periodic_refresh(self, interval_hours: int | None = None) -> None:
        interval = interval_hours or self.config.refresh_interval_hours
        sleep_seconds = max(300, int(interval * 3600))

        while True:
            try:
                await self.full_scan()
            except Exception:
                log.exception("Periodic scanner refresh failed")
            await asyncio.sleep(sleep_seconds)
