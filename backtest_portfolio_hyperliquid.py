#!/usr/bin/env python3
"""
Backtest portfolio-level de la strategie squeeze sur donnees Hyperliquid.

Modes:
- Backtest classique sur une periode (--start/--end optionnels)
- Walk-forward (train/test glissant) avec --walk-forward

Usage:
  python3 backtest_portfolio_hyperliquid.py --db squeeze_data.db
  python3 backtest_portfolio_hyperliquid.py --walk-forward --wf-train-days 120 --wf-test-days 30
"""

import argparse
from collections import defaultdict, deque
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd

from squeeze_detector import (
    BreakoutDirection,
    SqueezeConfig,
    SqueezePhase,
    compute_indicators,
    compute_squeeze_score,
    determine_phase,
    estimate_expected_move,
    predict_direction,
)


DAY_MS = 86_400_000


@dataclass
class PortfolioPosition:
    symbol: str
    side: BreakoutDirection
    entry_time_ms: int
    entry_price: float
    size_usd: float
    stop_price: float
    take_profit_price: float
    trailing_stop_price: float = 0.0
    trailing_activated: bool = False
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    score_at_entry: float = 0.0
    phase_at_entry: str = ""
    confidence_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    stop_atr_used: float = 0.0
    target_atr_used: float = 0.0
    trailing_stop_pct_used: float = 0.0
    trailing_activation_pct_used: float = 0.0
    max_holding_hours_used: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.side == BreakoutDirection.LONG


def parse_optional_date(value: Optional[str], end_of_day: bool = False) -> Optional[int]:
    if not value:
        return None
    base = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        base = base.replace(hour=23, minute=59, second=59)
    return int(base.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def resolve_dataset_range(db_path: Path, interval: str = "1h") -> tuple[Optional[int], Optional[int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM candles
            WHERE source = 'hyperliquid' AND interval = ?
            """,
            (interval,),
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]
    finally:
        conn.close()


def load_symbols(
    conn: sqlite3.Connection,
    interval: str,
    min_candles: int,
    min_volume: float,
    max_tokens: int,
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> list[str]:
    query = """
        SELECT c.symbol
        FROM candles c
        LEFT JOIN token_meta tm
          ON tm.source = c.source AND tm.symbol = c.symbol
        WHERE c.source = 'hyperliquid' AND c.interval = ?
    """
    params: list[object] = [interval]
    if start_ms is not None:
        query += " AND c.timestamp >= ?"
        params.append(start_ms)
    if end_ms is not None:
        query += " AND c.timestamp <= ?"
        params.append(end_ms)

    query += """
        GROUP BY c.symbol
        HAVING COUNT(*) >= ? AND COALESCE(MAX(tm.volume_24h), 0) >= ?
        ORDER BY COALESCE(MAX(tm.volume_24h), 0) DESC, COUNT(*) DESC
        LIMIT ?
    """
    params.extend([min_candles, min_volume, max_tokens])

    rows = conn.execute(query, params).fetchall()
    return [str(r[0]) for r in rows]


def load_symbol_df(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> pd.DataFrame:
    query = """
        SELECT timestamp as t, open, high, low, close, volume
        FROM candles
        WHERE source = 'hyperliquid' AND interval = ? AND symbol = ?
    """
    params: list[object] = [interval, symbol]
    if start_ms is not None:
        query += " AND timestamp >= ?"
        params.append(start_ms)
    if end_ms is not None:
        query += " AND timestamp <= ?"
        params.append(end_ms)
    query += " ORDER BY timestamp ASC"

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df


def close_position(
    pos: PortfolioPosition,
    price: float,
    ts_ms: int,
    reason: str,
    taker_fee: float,
    slippage_bps: float,
) -> dict:
    slippage = slippage_bps / 10000
    if pos.is_long:
        exit_price = price * (1 - slippage)
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
    else:
        exit_price = price * (1 + slippage)
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

    gross_pnl = pos.size_usd * pnl_pct
    fees = pos.size_usd * taker_fee * 2.0
    net_pnl = gross_pnl - fees

    return {
        "symbol": pos.symbol,
        "side": pos.side.value,
        "entry_time": datetime.fromtimestamp(pos.entry_time_ms / 1000, tz=timezone.utc).isoformat(),
        "exit_time": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
        "entry_ts_ms": pos.entry_time_ms,
        "exit_ts_ms": ts_ms,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "size_usd": pos.size_usd,
        "pnl_pct": pnl_pct,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "holding_hours": (ts_ms - pos.entry_time_ms) / 3600000,
        "reason": reason,
        "score_at_entry": pos.score_at_entry,
        "phase_at_entry": pos.phase_at_entry,
        "confidence_at_entry": pos.confidence_at_entry,
        "atr_at_entry": pos.atr_at_entry,
        "stop_atr_used": pos.stop_atr_used,
        "target_atr_used": pos.target_atr_used,
        "trailing_stop_pct_used": pos.trailing_stop_pct_used,
        "trailing_activation_pct_used": pos.trailing_activation_pct_used,
        "max_holding_hours_used": pos.max_holding_hours_used,
    }


def parse_float_list(raw: str, fallback: list[float]) -> list[float]:
    values: list[float] = []
    for chunk in (raw or "").split(","):
        part = chunk.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values if values else fallback


def clone_args(base: argparse.Namespace, overrides: dict[str, float]) -> argparse.Namespace:
    payload = vars(base).copy()
    payload.update(overrides)
    return argparse.Namespace(**payload)


def resolve_threshold(side_value: float, base_value: float) -> float:
    if isinstance(side_value, float) and math.isnan(side_value):
        return base_value
    return base_value if side_value < 0 else side_value


def resolve_risk_value(side_value: float, base_value: float) -> float:
    if isinstance(side_value, float) and math.isnan(side_value):
        return base_value
    return base_value if side_value <= 0 else side_value


def side_filters(
    args: argparse.Namespace,
    direction: BreakoutDirection,
) -> tuple[float, float, float, float]:
    if direction == BreakoutDirection.LONG:
        return (
            resolve_threshold(args.min_squeeze_score_long, args.min_squeeze_score),
            resolve_threshold(args.min_direction_confidence_long, args.min_direction_confidence),
            resolve_threshold(args.min_volume_ratio_long, args.min_volume_ratio),
            resolve_threshold(args.min_expected_move_pct_long, args.min_expected_move_pct),
        )
    return (
        resolve_threshold(args.min_squeeze_score_short, args.min_squeeze_score),
        resolve_threshold(args.min_direction_confidence_short, args.min_direction_confidence),
        resolve_threshold(args.min_volume_ratio_short, args.min_volume_ratio),
        resolve_threshold(args.min_expected_move_pct_short, args.min_expected_move_pct),
    )


def side_risk_params(
    args: argparse.Namespace,
    direction: BreakoutDirection,
) -> tuple[float, float, float, float, float]:
    if direction == BreakoutDirection.LONG:
        return (
            resolve_risk_value(args.stop_atr_long, args.stop_atr),
            resolve_risk_value(args.target_atr_long, args.target_atr),
            resolve_risk_value(args.trailing_stop_pct_long, args.trailing_stop_pct),
            resolve_risk_value(args.trailing_activation_pct_long, args.trailing_activation_pct),
            resolve_risk_value(args.max_holding_hours_long, args.max_holding_hours),
        )
    return (
        resolve_risk_value(args.stop_atr_short, args.stop_atr),
        resolve_risk_value(args.target_atr_short, args.target_atr),
        resolve_risk_value(args.trailing_stop_pct_short, args.trailing_stop_pct),
        resolve_risk_value(args.trailing_activation_pct_short, args.trailing_activation_pct),
        resolve_risk_value(args.max_holding_hours_short, args.max_holding_hours),
    )


def pattern_rules_passed(args: argparse.Namespace, row: pd.Series, expected_move: float) -> int:
    passed = 0
    rsi = float(row.get("rsi", 50.0) or 50.0)
    ema_spread = float(row.get("ema_spread", 0.0) or 0.0)
    ema_trend_slope = float(row.get("ema_trend_slope", 0.0) or 0.0)
    ret_8 = float(row.get("ret_8", 0.0) or 0.0)

    if rsi <= args.pattern_rsi_max:
        passed += 1
    if ema_spread <= args.pattern_ema_spread_max:
        passed += 1
    if ema_trend_slope <= args.pattern_ema_trend_slope_max:
        passed += 1
    if ret_8 <= args.pattern_ret8_max:
        passed += 1
    if args.pattern_expected_move_max > 0 and expected_move <= args.pattern_expected_move_max:
        passed += 1
    return passed


def apply_runtime_defaults(args: argparse.Namespace) -> None:
    defaults: dict[str, object] = {
        "interval": "1h",
        "entry_mode": "squeeze",
        "min_entry_score": 0.45,
        "breakout_min_vol_ratio": 1.2,
        "breakout_require_trend": False,
        "ema_cross_require_trend": False,
        "ema_cross_min_trend_slope": 0.0005,
        "macd_min_hist_pct": 0.0002,
        "macd_require_zero_line": False,
        "macd_require_trend": False,
        "macd_min_trend_slope": 0.0004,
        "rsi_revert_long_rsi": 35.0,
        "rsi_revert_short_rsi": 65.0,
        "rsi_revert_long_bb_pos_max": 0.20,
        "rsi_revert_short_bb_pos_min": 0.80,
        "rsi_revert_trend_filter": False,
        "rsi_revert_max_adverse_trend_slope": 0.0015,
        "min_squeeze_score_long": -1.0,
        "min_squeeze_score_short": -1.0,
        "min_direction_confidence_long": -1.0,
        "min_direction_confidence_short": -1.0,
        "min_volume_ratio_long": -1.0,
        "min_volume_ratio_short": -1.0,
        "min_expected_move_pct_long": -1.0,
        "min_expected_move_pct_short": -1.0,
        "stop_atr_long": 0.0,
        "stop_atr_short": 0.0,
        "target_atr_long": 0.0,
        "target_atr_short": 0.0,
        "trailing_stop_pct_long": 0.0,
        "trailing_stop_pct_short": 0.0,
        "trailing_activation_pct_long": 0.0,
        "trailing_activation_pct_short": 0.0,
        "max_holding_hours_long": 0.0,
        "max_holding_hours_short": 0.0,
        "enable_pattern_filter": False,
        "pattern_min_rules": 2,
        "pattern_rsi_max": 46.8,
        "pattern_ema_spread_max": -0.0013,
        "pattern_ema_trend_slope_max": -0.0008,
        "pattern_ret8_max": -0.0023,
        "pattern_expected_move_max": 0.089,
        "enable_dynamic_whitelist": False,
        "whitelist_lookback_days": 30,
        "whitelist_top_n": 12,
        "whitelist_min_trades": 3,
        "whitelist_score": "pnl",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            continue
        current = getattr(args, key)
        if isinstance(current, float) and math.isnan(current):
            setattr(args, key, value)


def score_train_result(res: dict, min_trades: int) -> float:
    if not res.get("ok"):
        return -1e12
    trades = float(res.get("total_trades", 0))
    ret = float(res.get("return_pct", 0.0))
    dd = abs(float(res.get("max_drawdown_pct", 0.0)))
    pf = float(res.get("profit_factor", 0.0))
    if math.isinf(pf):
        pf = 3.0
    pf_term = (pf - 1.0) * 10.0
    trades_penalty = max(0.0, float(min_trades) - trades) * 1.5
    return ret + pf_term - 0.25 * dd - trades_penalty


def cap_candidates(cands: list[dict], max_candidates: int) -> list[dict]:
    if max_candidates <= 0 or len(cands) <= max_candidates:
        return cands
    kept: list[dict] = []
    step = len(cands) / max_candidates
    used = set()
    for i in range(max_candidates):
        idx = min(len(cands) - 1, int(i * step))
        while idx in used and idx < len(cands) - 1:
            idx += 1
        if idx in used:
            continue
        used.add(idx)
        kept.append(cands[idx])
    return kept


def build_walk_forward_candidates(args: argparse.Namespace) -> list[dict]:
    if not args.wf_optimize:
        return [{}]

    min_squeeze_scores = parse_float_list(
        args.wf_grid_min_squeeze_score,
        [args.min_squeeze_score],
    )
    min_direction_confs = parse_float_list(
        args.wf_grid_min_direction_confidence,
        [args.min_direction_confidence],
    )
    stop_atrs = parse_float_list(args.wf_grid_stop_atr, [args.stop_atr])
    target_atrs = parse_float_list(args.wf_grid_target_atr, [args.target_atr])
    trailing_stops = parse_float_list(
        args.wf_grid_trailing_stop_pct,
        [args.trailing_stop_pct],
    )

    combos: list[dict] = []
    for mss, mdc, stop_atr, target_atr, trailing in product(
        min_squeeze_scores,
        min_direction_confs,
        stop_atrs,
        target_atrs,
        trailing_stops,
    ):
        combos.append(
            {
                "min_squeeze_score": float(mss),
                "min_direction_confidence": float(mdc),
                "stop_atr": float(stop_atr),
                "target_atr": float(target_atr),
                "trailing_stop_pct": float(trailing),
            }
        )

    # Toujours inclure le set de base
    base = {
        "min_squeeze_score": float(args.min_squeeze_score),
        "min_direction_confidence": float(args.min_direction_confidence),
        "stop_atr": float(args.stop_atr),
        "target_atr": float(args.target_atr),
        "trailing_stop_pct": float(args.trailing_stop_pct),
    }
    combos.insert(0, base)

    # Dédupliquer
    uniq: list[dict] = []
    seen = set()
    for c in combos:
        key = (
            round(c["min_squeeze_score"], 6),
            round(c["min_direction_confidence"], 6),
            round(c["stop_atr"], 6),
            round(c["target_atr"], 6),
            round(c["trailing_stop_pct"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return cap_candidates(uniq, args.wf_max_candidates)


def run_portfolio_backtest(
    args: argparse.Namespace,
    start_ms: Optional[int],
    end_ms: Optional[int],
    print_report: bool = True,
    export_trades_path: str = "",
    export_equity_path: str = "",
) -> dict:
    apply_runtime_defaults(args)
    conn = sqlite3.connect(str(Path(args.db)))

    symbols = load_symbols(
        conn,
        interval=args.interval,
        min_candles=args.min_candles,
        min_volume=args.min_volume,
        max_tokens=args.max_tokens,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if not symbols:
        return {
            "ok": False,
            "error": "Aucun token eligible.",
            "selected_symbols": 0,
            "tested_symbols": 0,
        }

    detector_cfg = SqueezeConfig(
        min_squeeze_score=args.detector_min_score,
        ready_squeeze_score=args.detector_ready_score,
        firing_score=args.detector_firing_score,
    )

    symbol_data: dict[str, dict] = {}
    all_timestamps: set[int] = set()
    min_needed = max(args.min_candles, args.warmup_bars + 1)

    for symbol in symbols:
        df = load_symbol_df(conn, symbol, args.interval, start_ms, end_ms)
        if len(df) < min_needed:
            continue
        ind = compute_indicators(df, detector_cfg)
        ts_to_i = {int(ts): i for i, ts in enumerate(ind["t"].astype(int).tolist())}
        all_timestamps.update(ts_to_i.keys())
        symbol_data[symbol] = {"df": ind, "ts_to_i": ts_to_i}

    conn.close()

    if not symbol_data:
        return {
            "ok": False,
            "error": "Aucune serie exploitable apres prechargement.",
            "selected_symbols": len(symbols),
            "tested_symbols": 0,
        }

    timeline = sorted(all_timestamps)
    tested_symbols = sorted(symbol_data.keys())

    positions: dict[str, PortfolioPosition] = {}
    phase_cache: dict[str, SqueezePhase] = {s: SqueezePhase.NO_SQUEEZE for s in tested_symbols}
    last_entry_sec: dict[str, float] = {}
    last_loss_sec: float = 0.0

    equity = args.initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    daily_pnl = 0.0
    daily_trades = 0
    current_day = None
    daily_loss_lock = False

    closed_trades: list[dict] = []
    equity_rows: list[dict] = []
    filter_stats = {
        "rows_seen": 0,
        "phase_ok": 0,
        "score_ok": 0,
        "direction_ok": 0,
        "confidence_ok": 0,
        "volume_ok": 0,
        "move_ok": 0,
        "trend_ok": 0,
        "pattern_ok": 0,
        "whitelist_ok": 0,
        "candidate_final": 0,
    }
    closed_for_whitelist: list[dict] = []

    whitelist_lookback_ms = int(max(1, args.whitelist_lookback_days) * DAY_MS)
    wl_events: deque[tuple[int, str, float]] = deque()
    wl_pnl: dict[str, float] = defaultdict(float)
    wl_trades: dict[str, int] = defaultdict(int)

    def evict_whitelist_events(now_ms: int) -> None:
        if not args.enable_dynamic_whitelist:
            return
        cutoff = now_ms - whitelist_lookback_ms
        while wl_events and wl_events[0][0] < cutoff:
            _, sym, pnl = wl_events.popleft()
            wl_pnl[sym] -= pnl
            wl_trades[sym] = max(0, wl_trades[sym] - 1)

    def compute_whitelist_symbols(now_ms: int) -> Optional[set[str]]:
        if not args.enable_dynamic_whitelist:
            return None
        evict_whitelist_events(now_ms)
        scored: list[tuple[float, int, float, str]] = []
        for sym in tested_symbols:
            n = wl_trades.get(sym, 0)
            if n < args.whitelist_min_trades:
                continue
            pnl = wl_pnl.get(sym, 0.0)
            score = pnl / n if args.whitelist_score == "expectancy" else pnl
            scored.append((score, n, pnl, sym))
        if not scored:
            return None
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        top_n = max(1, args.whitelist_top_n)
        return {x[3] for x in scored[:top_n]}

    def ingest_closed_for_whitelist() -> None:
        if not args.enable_dynamic_whitelist:
            closed_for_whitelist.clear()
            return
        for tr in closed_for_whitelist:
            ts = int(tr.get("exit_ts_ms", 0))
            sym = str(tr["symbol"])
            pnl = float(tr["net_pnl"])
            wl_events.append((ts, sym, pnl))
            wl_pnl[sym] += pnl
            wl_trades[sym] += 1
        closed_for_whitelist.clear()

    required_phases = {SqueezePhase.READY, SqueezePhase.FIRING}
    exposure_per_position = args.max_position_usd * args.leverage

    for ts_ms in timeline:
        ts_sec = ts_ms / 1000
        dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        day = dt.date()

        if current_day != day:
            current_day = day
            daily_pnl = 0.0
            daily_trades = 0
            daily_loss_lock = False

        # 1) Gestion des positions ouvertes
        to_close: list[tuple[str, float, str]] = []
        for symbol, pos in positions.items():
            state = symbol_data.get(symbol)
            if not state:
                continue
            i = state["ts_to_i"].get(ts_ms)
            if i is None:
                continue
            row = state["df"].iloc[i]
            price = float(row["close"])

            if pos.is_long:
                pos.highest_price = max(pos.highest_price, price)
                profit_pct = (price - pos.entry_price) / pos.entry_price
                if profit_pct >= pos.trailing_activation_pct_used:
                    pos.trailing_activated = True
                    new_trailing = pos.highest_price * (1.0 - pos.trailing_stop_pct_used)
                    pos.trailing_stop_price = max(pos.trailing_stop_price, new_trailing)
            else:
                pos.lowest_price = min(pos.lowest_price, price)
                profit_pct = (pos.entry_price - price) / pos.entry_price
                if profit_pct >= pos.trailing_activation_pct_used:
                    pos.trailing_activated = True
                    new_trailing = pos.lowest_price * (1.0 + pos.trailing_stop_pct_used)
                    if pos.trailing_stop_price == 0:
                        pos.trailing_stop_price = new_trailing
                    else:
                        pos.trailing_stop_price = min(pos.trailing_stop_price, new_trailing)

            hold_hours = (ts_ms - pos.entry_time_ms) / 3600000
            reason = ""
            if pos.is_long and price <= pos.stop_price:
                reason = "stop_loss"
            elif (not pos.is_long) and price >= pos.stop_price:
                reason = "stop_loss"
            elif pos.is_long and price >= pos.take_profit_price:
                reason = "take_profit"
            elif (not pos.is_long) and price <= pos.take_profit_price:
                reason = "take_profit"
            elif pos.trailing_activated:
                if pos.is_long and price <= pos.trailing_stop_price:
                    reason = "trailing_stop"
                elif (not pos.is_long) and price >= pos.trailing_stop_price:
                    reason = "trailing_stop"
            if not reason and hold_hours >= pos.max_holding_hours_used:
                reason = "max_time"

            if reason:
                to_close.append((symbol, price, reason))

        for symbol, price, reason in to_close:
            pos = positions.pop(symbol)
            trade = close_position(
                pos=pos,
                price=price,
                ts_ms=ts_ms,
                reason=reason,
                taker_fee=args.taker_fee,
                slippage_bps=args.slippage_bps,
            )
            closed_trades.append(trade)
            closed_for_whitelist.append(trade)
            equity += trade["net_pnl"]
            daily_pnl += trade["net_pnl"]
            if trade["net_pnl"] < 0:
                last_loss_sec = ts_sec

        # 2) Kill-switch journalier
        if (not daily_loss_lock) and daily_pnl <= -args.max_daily_loss_usd:
            daily_loss_lock = True
            force_close: list[tuple[str, float]] = []
            for symbol, pos in positions.items():
                state = symbol_data.get(symbol)
                if not state:
                    continue
                i = state["ts_to_i"].get(ts_ms)
                if i is None:
                    continue
                force_close.append((symbol, float(state["df"].iloc[i]["close"])))

            for symbol, price in force_close:
                pos = positions.pop(symbol)
                trade = close_position(
                    pos=pos,
                    price=price,
                    ts_ms=ts_ms,
                    reason="daily_loss_limit",
                    taker_fee=args.taker_fee,
                    slippage_bps=args.slippage_bps,
                )
                closed_trades.append(trade)
                closed_for_whitelist.append(trade)
                equity += trade["net_pnl"]
                daily_pnl += trade["net_pnl"]
                if trade["net_pnl"] < 0:
                    last_loss_sec = ts_sec

        # 3) Scan des entrees
        if (not daily_loss_lock) and daily_trades < args.max_trades_per_day:
            candidates: list[dict] = []
            allowed_symbols = compute_whitelist_symbols(ts_ms)

            for symbol in tested_symbols:
                if symbol in positions:
                    continue

                state = symbol_data[symbol]
                i = state["ts_to_i"].get(ts_ms)
                if i is None or i < args.warmup_bars:
                    continue

                row = state["df"].iloc[i]
                filter_stats["rows_seen"] += 1
                if allowed_symbols is not None and symbol not in allowed_symbols:
                    continue
                filter_stats["whitelist_ok"] += 1

                score = compute_squeeze_score(row, detector_cfg)
                direction = BreakoutDirection.UNKNOWN
                confidence = 0.0
                prev_phase = phase_cache[symbol]
                phase = SqueezePhase.NO_SQUEEZE
                expected_move = estimate_expected_move(row)
                vol_ratio = float(row.get("vol_ratio", 0.0) or 0.0)

                if args.entry_mode == "squeeze":
                    direction, confidence = predict_direction(row, detector_cfg)
                    phase = determine_phase(row, prev_phase, score, detector_cfg)
                    phase_cache[symbol] = phase

                    if phase not in required_phases:
                        continue
                    filter_stats["phase_ok"] += 1
                    if direction == BreakoutDirection.UNKNOWN:
                        continue
                    filter_stats["direction_ok"] += 1

                    min_score, min_conf_side, min_vol_ratio, min_expected_move = side_filters(args, direction)
                    if score < min_score:
                        continue
                    filter_stats["score_ok"] += 1

                    min_conf = min_conf_side
                    if phase == SqueezePhase.READY:
                        min_conf = max(min_conf, args.min_ready_confidence)
                    elif phase == SqueezePhase.FIRING:
                        min_conf = max(min_conf, args.min_firing_confidence)
                    if confidence < min_conf:
                        continue
                    filter_stats["confidence_ok"] += 1
                    if vol_ratio < min_vol_ratio:
                        continue
                    filter_stats["volume_ok"] += 1
                    if expected_move < min_expected_move:
                        continue
                    filter_stats["move_ok"] += 1

                elif args.entry_mode == "breakout":
                    is_up = bool(row.get("breakout_up", False))
                    is_down = bool(row.get("breakout_down", False))
                    if is_up == is_down:
                        continue
                    direction = BreakoutDirection.LONG if is_up else BreakoutDirection.SHORT
                    filter_stats["phase_ok"] += 1
                    filter_stats["direction_ok"] += 1

                    if vol_ratio < args.breakout_min_vol_ratio:
                        continue
                    filter_stats["volume_ok"] += 1
                    if expected_move < args.min_expected_move_pct:
                        continue
                    filter_stats["move_ok"] += 1

                    ret3 = abs(float(row.get("ret_3", 0.0) or 0.0))
                    confidence = float(
                        min(0.99, 0.50 + min(0.25, vol_ratio / 5.0) + min(0.20, ret3 * 20.0))
                    )
                    score = float(
                        min(1.0, 0.45 + min(0.25, ret3 * 25.0) + min(0.30, max(0.0, vol_ratio - 1.0)))
                    )
                    if score < args.min_entry_score:
                        continue
                    filter_stats["score_ok"] += 1
                    if confidence < args.min_direction_confidence:
                        continue
                    filter_stats["confidence_ok"] += 1

                elif args.entry_mode == "ema_cross":
                    if i <= 0:
                        continue
                    prev_row = state["df"].iloc[i - 1]
                    prev_spread = float(prev_row.get("ema_fast", 0.0) or 0.0) - float(
                        prev_row.get("ema_slow", 0.0) or 0.0
                    )
                    curr_spread = float(row.get("ema_fast", 0.0) or 0.0) - float(
                        row.get("ema_slow", 0.0) or 0.0
                    )
                    cross_up = prev_spread <= 0 and curr_spread > 0
                    cross_down = prev_spread >= 0 and curr_spread < 0
                    if not cross_up and not cross_down:
                        continue
                    direction = BreakoutDirection.LONG if cross_up else BreakoutDirection.SHORT
                    filter_stats["phase_ok"] += 1
                    filter_stats["direction_ok"] += 1

                    if vol_ratio < args.min_volume_ratio:
                        continue
                    filter_stats["volume_ok"] += 1
                    if expected_move < args.min_expected_move_pct:
                        continue
                    filter_stats["move_ok"] += 1

                    trend_slope = float(row.get("ema_trend_slope", 0.0) or 0.0)
                    if args.ema_cross_require_trend:
                        if direction == BreakoutDirection.LONG and trend_slope < args.ema_cross_min_trend_slope:
                            continue
                        if direction == BreakoutDirection.SHORT and trend_slope > -args.ema_cross_min_trend_slope:
                            continue

                    spread_norm = abs(float(row.get("ema_spread", 0.0) or 0.0))
                    confidence = float(min(0.99, 0.50 + min(0.35, spread_norm * 120.0)))
                    score = float(min(1.0, 0.45 + min(0.40, spread_norm * 150.0)))
                    if score < args.min_entry_score:
                        continue
                    filter_stats["score_ok"] += 1
                    if confidence < args.min_direction_confidence:
                        continue
                    filter_stats["confidence_ok"] += 1

                elif args.entry_mode == "macd":
                    if i <= 0:
                        continue
                    prev_row = state["df"].iloc[i - 1]
                    prev_macd = float(prev_row.get("macd", 0.0) or 0.0)
                    prev_signal = float(prev_row.get("macd_signal", 0.0) or 0.0)
                    curr_macd = float(row.get("macd", 0.0) or 0.0)
                    curr_signal = float(row.get("macd_signal", 0.0) or 0.0)
                    cross_up = prev_macd <= prev_signal and curr_macd > curr_signal
                    cross_down = prev_macd >= prev_signal and curr_macd < curr_signal
                    if not cross_up and not cross_down:
                        continue
                    direction = BreakoutDirection.LONG if cross_up else BreakoutDirection.SHORT
                    filter_stats["phase_ok"] += 1
                    filter_stats["direction_ok"] += 1

                    if vol_ratio < args.min_volume_ratio:
                        continue
                    filter_stats["volume_ok"] += 1
                    if expected_move < args.min_expected_move_pct:
                        continue
                    filter_stats["move_ok"] += 1

                    close_px = max(1e-9, float(row.get("close", 0.0) or 0.0))
                    hist = float(row.get("macd_hist", 0.0) or 0.0)
                    hist_slope = float(row.get("macd_hist_slope", 0.0) or 0.0)
                    hist_pct = abs(hist) / close_px
                    slope_pct = abs(hist_slope) / close_px

                    if hist_pct < args.macd_min_hist_pct:
                        continue
                    if args.macd_require_zero_line:
                        if direction == BreakoutDirection.LONG and curr_macd <= 0:
                            continue
                        if direction == BreakoutDirection.SHORT and curr_macd >= 0:
                            continue

                    trend_slope = float(row.get("ema_trend_slope", 0.0) or 0.0)
                    if args.macd_require_trend:
                        if direction == BreakoutDirection.LONG and trend_slope < args.macd_min_trend_slope:
                            continue
                        if direction == BreakoutDirection.SHORT and trend_slope > -args.macd_min_trend_slope:
                            continue

                    confidence = float(
                        min(0.99, 0.50 + min(0.30, hist_pct * 5000.0) + min(0.15, slope_pct * 7000.0))
                    )
                    score = float(min(1.0, 0.45 + min(0.35, hist_pct * 7000.0) + min(0.20, slope_pct * 9000.0)))
                    if score < args.min_entry_score:
                        continue
                    filter_stats["score_ok"] += 1
                    if confidence < args.min_direction_confidence:
                        continue
                    filter_stats["confidence_ok"] += 1

                elif args.entry_mode == "rsi_reversion":
                    rsi = float(row.get("rsi", 50.0) or 50.0)
                    bb_pos = float(row.get("bb_position", 0.5) or 0.5)
                    if rsi <= args.rsi_revert_long_rsi and bb_pos <= args.rsi_revert_long_bb_pos_max:
                        direction = BreakoutDirection.LONG
                        stretch = max(
                            (args.rsi_revert_long_rsi - rsi) / max(1.0, args.rsi_revert_long_rsi),
                            (args.rsi_revert_long_bb_pos_max - bb_pos) / max(0.05, args.rsi_revert_long_bb_pos_max),
                        )
                    elif rsi >= args.rsi_revert_short_rsi and bb_pos >= args.rsi_revert_short_bb_pos_min:
                        direction = BreakoutDirection.SHORT
                        stretch = max(
                            (rsi - args.rsi_revert_short_rsi) / max(1.0, 100.0 - args.rsi_revert_short_rsi),
                            (bb_pos - args.rsi_revert_short_bb_pos_min) / max(
                                0.05, 1.0 - args.rsi_revert_short_bb_pos_min
                            ),
                        )
                    else:
                        continue

                    filter_stats["phase_ok"] += 1
                    filter_stats["direction_ok"] += 1

                    if vol_ratio < args.min_volume_ratio:
                        continue
                    filter_stats["volume_ok"] += 1
                    if expected_move < args.min_expected_move_pct:
                        continue
                    filter_stats["move_ok"] += 1

                    if args.rsi_revert_trend_filter:
                        trend_slope = float(row.get("ema_trend_slope", 0.0) or 0.0)
                        limit = abs(args.rsi_revert_max_adverse_trend_slope)
                        if direction == BreakoutDirection.LONG and trend_slope < -limit:
                            continue
                        if direction == BreakoutDirection.SHORT and trend_slope > limit:
                            continue

                    confidence = float(min(0.99, 0.55 + min(0.30, stretch)))
                    score = float(min(1.0, 0.50 + min(0.35, stretch)))
                    if score < args.min_entry_score:
                        continue
                    filter_stats["score_ok"] += 1
                    if confidence < args.min_direction_confidence:
                        continue
                    filter_stats["confidence_ok"] += 1
                else:
                    continue

                ema_trend = float(row.get("ema_trend", 0.0) or 0.0)
                close_price = float(row["close"])
                if args.entry_mode == "rsi_reversion":
                    trend_ok = True
                elif args.entry_mode == "breakout" and (not args.breakout_require_trend):
                    trend_ok = True
                elif args.entry_mode == "ema_cross" and (not args.ema_cross_require_trend):
                    trend_ok = True
                elif args.entry_mode == "macd" and (not args.macd_require_trend):
                    trend_ok = True
                else:
                    trend_ok = (
                        (direction == BreakoutDirection.LONG and close_price > ema_trend)
                        or (direction == BreakoutDirection.SHORT and close_price < ema_trend)
                    )
                if not trend_ok:
                    continue
                filter_stats["trend_ok"] += 1

                if args.enable_pattern_filter:
                    rules_passed = pattern_rules_passed(args, row, expected_move)
                    if rules_passed < args.pattern_min_rules:
                        continue
                filter_stats["pattern_ok"] += 1

                stop_atr, target_atr, trailing_stop_pct, trailing_activation_pct, max_holding_hours = side_risk_params(
                    args,
                    direction,
                )
                candidates.append(
                    {
                        "symbol": symbol,
                        "row": row,
                        "score": score,
                        "direction": direction,
                        "confidence": confidence,
                        "phase": phase,
                        "stop_atr": stop_atr,
                        "target_atr": target_atr,
                        "trailing_stop_pct": trailing_stop_pct,
                        "trailing_activation_pct": trailing_activation_pct,
                        "max_holding_hours": max_holding_hours,
                    }
                )
                filter_stats["candidate_final"] += 1

            candidates.sort(key=lambda c: (c["score"], c["confidence"]), reverse=True)

            for c in candidates:
                symbol = c["symbol"]
                if symbol in positions:
                    continue
                if len(positions) >= args.max_positions:
                    break

                if ts_sec - last_entry_sec.get(symbol, 0.0) < args.entry_cooldown_minutes * 60:
                    continue
                if last_loss_sec > 0 and ts_sec - last_loss_sec < args.cooldown_after_loss_sec:
                    continue

                current_exposure = sum(p.size_usd for p in positions.values())
                if current_exposure + exposure_per_position > args.max_total_exposure_usd:
                    continue

                row = c["row"]
                price = float(row["close"])
                atr = float(row.get("atr", 0.0) or 0.0)
                if atr <= 0:
                    atr = price * 0.02

                slippage = args.slippage_bps / 10000
                if c["direction"] == BreakoutDirection.LONG:
                    entry_price = price * (1 + slippage)
                    stop_price = entry_price - c["stop_atr"] * atr
                    target_price = entry_price + c["target_atr"] * atr
                else:
                    entry_price = price * (1 - slippage)
                    stop_price = entry_price + c["stop_atr"] * atr
                    target_price = entry_price - c["target_atr"] * atr

                positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    side=c["direction"],
                    entry_time_ms=ts_ms,
                    entry_price=entry_price,
                    size_usd=exposure_per_position,
                    stop_price=stop_price,
                    take_profit_price=target_price,
                    highest_price=entry_price,
                    lowest_price=entry_price,
                    score_at_entry=float(c["score"]),
                    phase_at_entry=c["phase"].value,
                    confidence_at_entry=float(c["confidence"]),
                    atr_at_entry=atr,
                    stop_atr_used=float(c["stop_atr"]),
                    target_atr_used=float(c["target_atr"]),
                    trailing_stop_pct_used=float(c["trailing_stop_pct"]),
                    trailing_activation_pct_used=float(c["trailing_activation_pct"]),
                    max_holding_hours_used=float(c["max_holding_hours"]),
                )
                last_entry_sec[symbol] = ts_sec
                daily_trades += 1

                if daily_trades >= args.max_trades_per_day:
                    break

        # 3b) Mise a jour whitelist apres le scan de la barre courante (pas de look-ahead)
        ingest_closed_for_whitelist()

        # 4) Equity curve MTM
        unrealized = 0.0
        for symbol, pos in positions.items():
            state = symbol_data[symbol]
            i = state["ts_to_i"].get(ts_ms)
            if i is None:
                continue
            px = float(state["df"].iloc[i]["close"])
            pnl_pct = (px - pos.entry_price) / pos.entry_price if pos.is_long else (pos.entry_price - px) / pos.entry_price
            unrealized += pos.size_usd * pnl_pct

        mtm_equity = equity + unrealized
        peak_equity = max(peak_equity, mtm_equity)
        drawdown = (mtm_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        equity_rows.append(
            {
                "timestamp": dt.isoformat(),
                "equity_realized": equity,
                "equity_mtm": mtm_equity,
                "open_positions": len(positions),
                "daily_pnl": daily_pnl,
            }
        )

    # Cloture finale forcee
    if timeline and positions:
        final_ts = timeline[-1]
        for symbol in list(positions.keys()):
            pos = positions.pop(symbol)
            state = symbol_data[symbol]
            i = state["ts_to_i"].get(final_ts)
            if i is None:
                continue
            px = float(state["df"].iloc[i]["close"])
            trade = close_position(
                pos=pos,
                price=px,
                ts_ms=final_ts,
                reason="end_of_test",
                taker_fee=args.taker_fee,
                slippage_bps=args.slippage_bps,
            )
            closed_trades.append(trade)
            equity += trade["net_pnl"]

    trades_df = pd.DataFrame(closed_trades)
    equity_df = pd.DataFrame(equity_rows)

    total_trades = len(trades_df)
    total_pnl = float(trades_df["net_pnl"].sum()) if total_trades else 0.0
    final_equity = args.initial_capital + total_pnl
    return_pct = (final_equity / args.initial_capital - 1.0) * 100 if args.initial_capital > 0 else 0.0
    win_rate = float((trades_df["net_pnl"] > 0).mean() * 100) if total_trades else 0.0
    gross_win = float(trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].sum()) if total_trades else 0.0
    gross_loss = float(trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].sum()) if total_trades else 0.0
    profit_factor = abs(gross_win / gross_loss) if gross_loss < 0 else (float("inf") if gross_win > 0 else 0.0)

    result = {
        "ok": True,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "selected_symbols": len(symbols),
        "tested_symbols": len(tested_symbols),
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown * 100,
        "trades_df": trades_df,
        "equity_df": equity_df,
        "filter_stats": filter_stats,
    }

    if print_report:
        print("=" * 95)
        print("PORTFOLIO BACKTEST - SQUEEZE HYPERLIQUID")
        print("=" * 95)
        print(f"DB: {args.db}")
        print(f"Interval: {args.interval}")
        print(f"Entry mode: {args.entry_mode}")
        print(
            f"Periode: {ms_to_iso(start_ms) if start_ms else 'debut'} -> "
            f"{ms_to_iso(end_ms) if end_ms else 'fin'}"
        )
        print(f"Tokens utilises: {len(tested_symbols)} / {len(symbols)} selectionnes")
        print("-" * 95)
        print(f"Capital initial:      ${args.initial_capital:,.2f}")
        print(f"Capital final:        ${final_equity:,.2f}")
        print(f"PnL total:            ${total_pnl:+,.2f} ({return_pct:+.2f}%)")
        print(f"Nombre de trades:     {total_trades}")
        print(f"Win rate:             {win_rate:.1f}%")
        print(f"Profit factor:        {profit_factor:.2f}")
        print(f"Max drawdown (MTM):   {max_drawdown * 100:.2f}%")
        print("-" * 95)

        if total_trades:
            by_reason = (
                trades_df.groupby("reason")["net_pnl"]
                .agg(["count", "sum"])
                .sort_values("sum", ascending=False)
            )
            print("Sorties par raison:")
            print(by_reason.to_string())
            print("-" * 95)
        else:
            print("Funnel filtres (debug):")
            for k, v in filter_stats.items():
                print(f"  {k}: {v}")
            print("-" * 95)

    if export_trades_path:
        trades_df.to_csv(Path(export_trades_path), index=False)
        if print_report:
            print(f"Trades exportes: {export_trades_path}")

    if export_equity_path:
        equity_df.to_csv(Path(export_equity_path), index=False)
        if print_report:
            print(f"Equity curve exportee: {export_equity_path}")

    return result


def run_walk_forward(args: argparse.Namespace, user_start_ms: Optional[int], user_end_ms: Optional[int]) -> int:
    ds_start, ds_end = resolve_dataset_range(Path(args.db), args.interval)
    if ds_start is None or ds_end is None:
        print("Impossible de lire la plage temporelle de la base.")
        return 1

    range_start = user_start_ms if user_start_ms is not None else ds_start
    range_end = user_end_ms if user_end_ms is not None else ds_end
    if range_start >= range_end:
        print("Plage temporelle invalide pour walk-forward.")
        return 1

    train_ms = args.wf_train_days * DAY_MS
    test_ms = args.wf_test_days * DAY_MS
    step_ms = args.wf_step_days * DAY_MS

    if train_ms <= 0 or test_ms <= 0 or step_ms <= 0:
        print("wf-train-days, wf-test-days et wf-step-days doivent etre > 0")
        return 1

    candidates = build_walk_forward_candidates(args)

    cursor = range_start
    window = 1
    rows: list[dict] = []

    print("=" * 105)
    print("WALK-FORWARD PORTFOLIO BACKTEST")
    print("=" * 105)
    print(
        f"Range: {ms_to_iso(range_start)} -> {ms_to_iso(range_end)} | "
        f"train={args.wf_train_days}j test={args.wf_test_days}j step={args.wf_step_days}j"
    )
    if args.wf_optimize:
        print(
            f"Optimisation train-only active | "
            f"candidats={len(candidates)} | min_train_trades={args.wf_min_train_trades}"
        )
    print("-" * 105)

    while True:
        train_start = cursor
        train_end = train_start + train_ms - 1
        test_start = train_end + 1
        test_end = test_start + test_ms - 1

        if test_end > range_end:
            break

        if args.wf_optimize:
            best_score = -1e12
            best_params = candidates[0]
            best_train_res = {"ok": False}
            for cand in candidates:
                tuned_args = clone_args(args, cand)
                train_res_candidate = run_portfolio_backtest(
                    tuned_args,
                    start_ms=train_start,
                    end_ms=train_end,
                    print_report=False,
                )
                score = score_train_result(train_res_candidate, args.wf_min_train_trades)
                if score > best_score:
                    best_score = score
                    best_params = cand
                    best_train_res = train_res_candidate

            best_args = clone_args(args, best_params)
            train_res = best_train_res
            test_res = run_portfolio_backtest(
                best_args,
                start_ms=test_start,
                end_ms=test_end,
                print_report=False,
            )
        else:
            best_score = 0.0
            best_params = {}
            train_res = run_portfolio_backtest(
                args,
                start_ms=train_start,
                end_ms=train_end,
                print_report=False,
            )
            test_res = run_portfolio_backtest(
                args,
                start_ms=test_start,
                end_ms=test_end,
                print_report=False,
            )

        row = {
            "window": window,
            "train_start": ms_to_iso(train_start),
            "train_end": ms_to_iso(train_end),
            "test_start": ms_to_iso(test_start),
            "test_end": ms_to_iso(test_end),
            "train_return_pct": train_res.get("return_pct", 0.0) if train_res.get("ok") else 0.0,
            "train_trades": train_res.get("total_trades", 0) if train_res.get("ok") else 0,
            "test_return_pct": test_res.get("return_pct", 0.0) if test_res.get("ok") else 0.0,
            "test_trades": test_res.get("total_trades", 0) if test_res.get("ok") else 0,
            "test_win_rate": test_res.get("win_rate", 0.0) if test_res.get("ok") else 0.0,
            "test_profit_factor": test_res.get("profit_factor", 0.0) if test_res.get("ok") else 0.0,
            "test_max_dd_pct": test_res.get("max_drawdown_pct", 0.0) if test_res.get("ok") else 0.0,
            "test_ok": bool(test_res.get("ok")),
            "train_score": best_score,
            "best_min_squeeze_score": best_params.get("min_squeeze_score", args.min_squeeze_score),
            "best_min_direction_confidence": best_params.get("min_direction_confidence", args.min_direction_confidence),
            "best_stop_atr": best_params.get("stop_atr", args.stop_atr),
            "best_target_atr": best_params.get("target_atr", args.target_atr),
            "best_trailing_stop_pct": best_params.get("trailing_stop_pct", args.trailing_stop_pct),
        }
        rows.append(row)

        if args.wf_optimize:
            print(
                f"W{window:02d} | Test {row['test_start']} -> {row['test_end']} | "
                f"ret={row['test_return_pct']:+6.2f}% | trades={row['test_trades']:4d} | "
                f"pf={row['test_profit_factor']:.2f} | "
                f"best(mss={row['best_min_squeeze_score']:.2f}, conf={row['best_min_direction_confidence']:.2f}, "
                f"sl={row['best_stop_atr']:.2f}, tp={row['best_target_atr']:.2f}, tr={row['best_trailing_stop_pct']:.3f})"
            )
        else:
            print(
                f"W{window:02d} | Test {row['test_start']} -> {row['test_end']} | "
                f"ret={row['test_return_pct']:+6.2f}% | trades={row['test_trades']:4d} | "
                f"pf={row['test_profit_factor']:.2f}"
            )

        window += 1
        cursor += step_ms

    if not rows:
        print("Aucune fenetre walk-forward generee avec ces parametres.")
        return 1

    wf_df = pd.DataFrame(rows)
    valid = wf_df[wf_df["test_ok"] == True].copy()

    if valid.empty:
        print("Aucune fenetre test valide.")
        return 1

    compounded = args.initial_capital
    for r in valid["test_return_pct"].tolist():
        compounded *= (1.0 + r / 100.0)
    compounded_return_pct = (compounded / args.initial_capital - 1.0) * 100.0

    avg_test_return = valid["test_return_pct"].mean()
    median_test_return = valid["test_return_pct"].median()
    avg_pf = valid["test_profit_factor"].mean()
    total_test_trades = int(valid["test_trades"].sum())
    positive_windows = int((valid["test_return_pct"] > 0).sum())

    print("-" * 105)
    print(f"Windows valides: {len(valid)}/{len(wf_df)}")
    print(f"Test return moyen: {avg_test_return:+.2f}% | median: {median_test_return:+.2f}%")
    print(f"PF moyen (test): {avg_pf:.2f}")
    print(f"Total trades (test): {total_test_trades}")
    print(f"Fenetre test positives: {positive_windows}/{len(valid)}")
    print(f"Return compose (OOS): {compounded_return_pct:+.2f}%")
    print("=" * 105)

    if args.export_wf:
        wf_df.to_csv(Path(args.export_wf), index=False)
        print(f"Walk-forward exporte: {args.export_wf}")

    return 0


def run() -> int:
    parser = argparse.ArgumentParser(description="Portfolio-level backtest for Hyperliquid squeeze strategy")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--min-candles", type=int, default=500)
    parser.add_argument("--min-volume", type=float, default=100_000)

    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--max-position-usd", type=float, default=30.0)
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--max-total-exposure-usd", type=float, default=200.0)
    parser.add_argument("--leverage", type=float, default=3.0)

    parser.add_argument("--min-squeeze-score", type=float, default=0.55)
    parser.add_argument("--min-direction-confidence", type=float, default=0.60)
    parser.add_argument("--min-ready-confidence", type=float, default=0.62)
    parser.add_argument("--min-firing-confidence", type=float, default=0.55)
    parser.add_argument("--min-volume-ratio", type=float, default=0.30)
    parser.add_argument("--min-expected-move-pct", type=float, default=0.02)
    parser.add_argument(
        "--entry-mode",
        type=str,
        choices=["squeeze", "breakout", "ema_cross", "macd", "rsi_reversion"],
        default="squeeze",
    )
    parser.add_argument("--min-entry-score", type=float, default=0.45)
    parser.add_argument("--breakout-min-vol-ratio", type=float, default=1.2)
    parser.add_argument("--breakout-require-trend", action="store_true")
    parser.add_argument("--ema-cross-require-trend", action="store_true")
    parser.add_argument("--ema-cross-min-trend-slope", type=float, default=0.0005)
    parser.add_argument("--macd-min-hist-pct", type=float, default=0.0002)
    parser.add_argument("--macd-require-zero-line", action="store_true")
    parser.add_argument("--macd-require-trend", action="store_true")
    parser.add_argument("--macd-min-trend-slope", type=float, default=0.0004)
    parser.add_argument("--rsi-revert-long-rsi", type=float, default=35.0)
    parser.add_argument("--rsi-revert-short-rsi", type=float, default=65.0)
    parser.add_argument("--rsi-revert-long-bb-pos-max", type=float, default=0.20)
    parser.add_argument("--rsi-revert-short-bb-pos-min", type=float, default=0.80)
    parser.add_argument("--rsi-revert-trend-filter", action="store_true")
    parser.add_argument("--rsi-revert-max-adverse-trend-slope", type=float, default=0.0015)
    parser.add_argument("--min-squeeze-score-long", type=float, default=-1.0)
    parser.add_argument("--min-squeeze-score-short", type=float, default=-1.0)
    parser.add_argument("--min-direction-confidence-long", type=float, default=-1.0)
    parser.add_argument("--min-direction-confidence-short", type=float, default=-1.0)
    parser.add_argument("--min-volume-ratio-long", type=float, default=-1.0)
    parser.add_argument("--min-volume-ratio-short", type=float, default=-1.0)
    parser.add_argument("--min-expected-move-pct-long", type=float, default=-1.0)
    parser.add_argument("--min-expected-move-pct-short", type=float, default=-1.0)
    parser.add_argument("--entry-cooldown-minutes", type=float, default=60.0)
    parser.add_argument("--cooldown-after-loss-sec", type=float, default=300.0)

    parser.add_argument("--max-daily-loss-usd", type=float, default=15.0)
    parser.add_argument("--max-trades-per-day", type=int, default=10)

    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--target-atr", type=float, default=3.0)
    parser.add_argument("--trailing-stop-pct", type=float, default=0.015)
    parser.add_argument("--trailing-activation-pct", type=float, default=0.01)
    parser.add_argument("--max-holding-hours", type=float, default=24.0)
    parser.add_argument("--stop-atr-long", type=float, default=0.0)
    parser.add_argument("--stop-atr-short", type=float, default=0.0)
    parser.add_argument("--target-atr-long", type=float, default=0.0)
    parser.add_argument("--target-atr-short", type=float, default=0.0)
    parser.add_argument("--trailing-stop-pct-long", type=float, default=0.0)
    parser.add_argument("--trailing-stop-pct-short", type=float, default=0.0)
    parser.add_argument("--trailing-activation-pct-long", type=float, default=0.0)
    parser.add_argument("--trailing-activation-pct-short", type=float, default=0.0)
    parser.add_argument("--max-holding-hours-long", type=float, default=0.0)
    parser.add_argument("--max-holding-hours-short", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--taker-fee", type=float, default=0.00035)

    parser.add_argument("--enable-pattern-filter", action="store_true")
    parser.add_argument("--pattern-min-rules", type=int, default=2)
    parser.add_argument("--pattern-rsi-max", type=float, default=46.8)
    parser.add_argument("--pattern-ema-spread-max", type=float, default=-0.0013)
    parser.add_argument("--pattern-ema-trend-slope-max", type=float, default=-0.0008)
    parser.add_argument("--pattern-ret8-max", type=float, default=-0.0023)
    parser.add_argument("--pattern-expected-move-max", type=float, default=0.089)

    parser.add_argument("--enable-dynamic-whitelist", action="store_true")
    parser.add_argument("--whitelist-lookback-days", type=int, default=30)
    parser.add_argument("--whitelist-top-n", type=int, default=12)
    parser.add_argument("--whitelist-min-trades", type=int, default=3)
    parser.add_argument("--whitelist-score", type=str, choices=["pnl", "expectancy"], default="pnl")

    parser.add_argument("--detector-min-score", type=float, default=0.45)
    parser.add_argument("--detector-ready-score", type=float, default=0.70)
    parser.add_argument("--detector-firing-score", type=float, default=0.50)
    parser.add_argument("--warmup-bars", type=int, default=100)

    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wf-train-days", type=int, default=120)
    parser.add_argument("--wf-test-days", type=int, default=30)
    parser.add_argument("--wf-step-days", type=int, default=30)
    parser.add_argument("--wf-optimize", action="store_true")
    parser.add_argument("--wf-min-train-trades", type=int, default=20)
    parser.add_argument("--wf-max-candidates", type=int, default=40)
    parser.add_argument("--wf-grid-min-squeeze-score", type=str, default="0.50,0.55,0.60")
    parser.add_argument("--wf-grid-min-direction-confidence", type=str, default="0.55,0.60,0.65")
    parser.add_argument("--wf-grid-stop-atr", type=str, default="1.3,1.5,1.7")
    parser.add_argument("--wf-grid-target-atr", type=str, default="2.5,3.0,3.5")
    parser.add_argument("--wf-grid-trailing-stop-pct", type=str, default="0.012,0.015,0.020")

    parser.add_argument("--export-trades", type=str, default="")
    parser.add_argument("--export-equity", type=str, default="")
    parser.add_argument("--export-wf", type=str, default="")

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB introuvable: {db_path}")
        return 1

    if args.max_total_exposure_usd < args.max_position_usd * args.leverage:
        print(
            "Warning: max_total_exposure_usd < exposition d'une position "
            f"({args.max_position_usd * args.leverage:.2f})."
        )
    if args.pattern_min_rules < 1:
        print("Erreur: --pattern-min-rules doit etre >= 1")
        return 1
    if args.whitelist_lookback_days < 1:
        print("Erreur: --whitelist-lookback-days doit etre >= 1")
        return 1
    if args.whitelist_top_n < 1:
        print("Erreur: --whitelist-top-n doit etre >= 1")
        return 1
    if args.whitelist_min_trades < 0:
        print("Erreur: --whitelist-min-trades doit etre >= 0")
        return 1

    start_ms = parse_optional_date(args.start)
    end_ms = parse_optional_date(args.end, end_of_day=True)
    if start_ms and end_ms and start_ms > end_ms:
        print("Erreur: --start doit etre <= --end")
        return 1

    if args.walk_forward:
        if args.export_trades or args.export_equity:
            print("Note: --export-trades/--export-equity ignores en mode walk-forward.")
        return run_walk_forward(args, start_ms, end_ms)

    result = run_portfolio_backtest(
        args,
        start_ms=start_ms,
        end_ms=end_ms,
        print_report=True,
        export_trades_path=args.export_trades,
        export_equity_path=args.export_equity,
    )
    if not result.get("ok"):
        print(result.get("error", "Erreur backtest"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
