#!/usr/bin/env python3
"""
Benchmark multi-timeframes et multi-strategies sur le moteur portfolio.

Exemples:
  python3 benchmark_timeframes_strategies.py --db squeeze_data.db --intervals 15m,30m,1h
  python3 benchmark_timeframes_strategies.py --window-mode common --export benchmark_common.csv
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest_portfolio_hyperliquid import (
    clone_args,
    parse_optional_date,
    resolve_dataset_range,
    run_portfolio_backtest,
)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def make_base_args(db: str, interval: str, max_tokens: int, min_candles: int) -> argparse.Namespace:
    return argparse.Namespace(
        db=db,
        interval=interval,
        start="",
        end="",
        max_tokens=max_tokens,
        min_candles=min_candles,
        min_volume=100_000,
        initial_capital=1000.0,
        max_position_usd=30.0,
        max_positions=2,
        max_total_exposure_usd=200.0,
        leverage=3.0,
        min_squeeze_score=0.60,
        min_direction_confidence=0.50,
        min_ready_confidence=0.62,
        min_firing_confidence=0.55,
        min_volume_ratio=0.25,
        min_expected_move_pct=0.02,
        entry_cooldown_minutes=60.0,
        cooldown_after_loss_sec=300.0,
        max_daily_loss_usd=15.0,
        max_trades_per_day=6,
        stop_atr=1.7,
        target_atr=2.3,
        trailing_stop_pct=0.020,
        trailing_activation_pct=0.008,
        max_holding_hours=24.0,
        slippage_bps=3.0,
        taker_fee=0.00035,
        detector_min_score=0.45,
        detector_ready_score=0.70,
        detector_firing_score=0.50,
        warmup_bars=100,
    )


def strategy_presets() -> list[tuple[str, dict]]:
    return [
        (
            "baseline",
            {
                "entry_mode": "squeeze",
                "enable_pattern_filter": False,
            },
        ),
        (
            "pattern_robust",
            {
                "entry_mode": "squeeze",
                "min_squeeze_score": 0.60,
                "min_direction_confidence": 0.50,
                "min_volume_ratio": 0.20,
                "stop_atr": 2.0,
                "target_atr": 2.1,
                "trailing_stop_pct": 0.020,
                "trailing_activation_pct": 0.008,
                "max_trades_per_day": 6,
                "enable_pattern_filter": True,
                "pattern_min_rules": 3,
                "pattern_rsi_max": 48.0,
                "pattern_ema_spread_max": -0.0030,
                "pattern_ema_trend_slope_max": -0.0018,
                "pattern_ret8_max": -0.0035,
                "pattern_expected_move_max": 0.100,
            },
        ),
        (
            "pattern_aggressive",
            {
                "entry_mode": "squeeze",
                "min_squeeze_score": 0.60,
                "min_direction_confidence": 0.40,
                "min_volume_ratio": 0.25,
                "stop_atr": 2.0,
                "target_atr": 2.1,
                "trailing_stop_pct": 0.020,
                "trailing_activation_pct": 0.008,
                "max_trades_per_day": 6,
                "enable_pattern_filter": True,
                "pattern_min_rules": 3,
                "pattern_rsi_max": 50.0,
                "pattern_ema_spread_max": -0.0018,
                "pattern_ema_trend_slope_max": -0.0018,
                "pattern_ret8_max": -0.0035,
                "pattern_expected_move_max": 0.089,
            },
        ),
        (
            "trend_follow",
            {
                "entry_mode": "squeeze",
                "min_squeeze_score": 0.65,
                "min_direction_confidence": 0.60,
                "min_volume_ratio": 0.30,
                "min_expected_move_pct": 0.03,
                "stop_atr": 1.6,
                "target_atr": 2.8,
                "trailing_stop_pct": 0.018,
                "trailing_activation_pct": 0.010,
                "max_trades_per_day": 4,
                "enable_pattern_filter": False,
            },
        ),
        (
            "mean_revert_squeeze",
            {
                "entry_mode": "squeeze",
                "min_squeeze_score": 0.55,
                "min_direction_confidence": 0.45,
                "min_volume_ratio": 0.20,
                "stop_atr": 1.8,
                "target_atr": 2.0,
                "trailing_stop_pct": 0.022,
                "trailing_activation_pct": 0.006,
                "max_trades_per_day": 6,
                "enable_pattern_filter": True,
                "pattern_min_rules": 4,
                "pattern_rsi_max": 46.0,
                "pattern_ema_spread_max": -0.0025,
                "pattern_ema_trend_slope_max": -0.0018,
                "pattern_ret8_max": -0.0035,
                "pattern_expected_move_max": 0.089,
            },
        ),
        (
            "split_long_short",
            {
                "entry_mode": "squeeze",
                "enable_pattern_filter": True,
                "pattern_min_rules": 2,
                "min_squeeze_score_long": 0.65,
                "min_squeeze_score_short": 0.55,
                "min_direction_confidence_long": 0.60,
                "min_direction_confidence_short": 0.45,
                "min_volume_ratio_long": 0.30,
                "min_volume_ratio_short": 0.20,
                "min_expected_move_pct_long": 0.025,
                "min_expected_move_pct_short": 0.015,
                "stop_atr_long": 1.8,
                "stop_atr_short": 1.6,
                "target_atr_long": 2.5,
                "target_atr_short": 2.1,
                "trailing_stop_pct_long": 0.020,
                "trailing_stop_pct_short": 0.018,
                "trailing_activation_pct_long": 0.010,
                "trailing_activation_pct_short": 0.007,
            },
        ),
        (
            "breakout_momentum",
            {
                "entry_mode": "breakout",
                "min_entry_score": 0.45,
                "min_direction_confidence": 0.50,
                "breakout_min_vol_ratio": 1.20,
                "breakout_require_trend": True,
                "min_volume_ratio": 0.20,
                "min_expected_move_pct": 0.02,
                "stop_atr": 1.6,
                "target_atr": 2.6,
                "trailing_stop_pct": 0.018,
                "trailing_activation_pct": 0.010,
                "max_trades_per_day": 5,
                "enable_pattern_filter": False,
            },
        ),
        (
            "ema_cross_trend",
            {
                "entry_mode": "ema_cross",
                "min_entry_score": 0.45,
                "min_direction_confidence": 0.50,
                "min_volume_ratio": 0.20,
                "min_expected_move_pct": 0.015,
                "ema_cross_require_trend": True,
                "ema_cross_min_trend_slope": 0.0005,
                "stop_atr": 1.5,
                "target_atr": 2.4,
                "trailing_stop_pct": 0.018,
                "trailing_activation_pct": 0.008,
                "max_trades_per_day": 6,
                "enable_pattern_filter": False,
            },
        ),
        (
            "rsi_reversion_classic",
            {
                "entry_mode": "rsi_reversion",
                "min_entry_score": 0.45,
                "min_direction_confidence": 0.50,
                "min_volume_ratio": 0.18,
                "min_expected_move_pct": 0.01,
                "rsi_revert_long_rsi": 33.0,
                "rsi_revert_short_rsi": 67.0,
                "rsi_revert_long_bb_pos_max": 0.18,
                "rsi_revert_short_bb_pos_min": 0.82,
                "rsi_revert_trend_filter": True,
                "rsi_revert_max_adverse_trend_slope": 0.0012,
                "stop_atr": 1.8,
                "target_atr": 2.0,
                "trailing_stop_pct": 0.020,
                "trailing_activation_pct": 0.006,
                "max_trades_per_day": 6,
                "enable_pattern_filter": False,
            },
        ),
    ]


def clamp_range(start_ms: int, end_ms: int, user_start: Optional[int], user_end: Optional[int]) -> tuple[int, int]:
    s = max(start_ms, user_start) if user_start is not None else start_ms
    e = min(end_ms, user_end) if user_end is not None else end_ms
    return s, e


def run() -> int:
    parser = argparse.ArgumentParser(description="Benchmark strategy x timeframe")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--intervals", type=str, default="15m,30m,1h")
    parser.add_argument("--window-mode", type=str, choices=["common", "full", "both"], default="both")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument("--min-candles", type=int, default=200)
    parser.add_argument("--export", type=str, default="strategy_timeframe_benchmark.csv")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB introuvable: {db_path}")
        return 1

    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]
    if not intervals:
        print("Aucun intervalle fourni.")
        return 1

    ranges: dict[str, tuple[int, int]] = {}
    for itv in intervals:
        s, e = resolve_dataset_range(db_path, itv)
        if s is None or e is None:
            print(f"Aucune donnee pour interval={itv}")
            continue
        ranges[itv] = (s, e)
    if not ranges:
        print("Aucune plage valide.")
        return 1

    user_start = parse_optional_date(args.start)
    user_end = parse_optional_date(args.end, end_of_day=True)

    common_start = max(v[0] for v in ranges.values())
    common_end = min(v[1] for v in ranges.values())
    if user_start is not None:
        common_start = max(common_start, user_start)
    if user_end is not None:
        common_end = min(common_end, user_end)

    modes: list[str]
    if args.window_mode == "both":
        modes = ["common", "full"]
    else:
        modes = [args.window_mode]

    strategies = strategy_presets()
    rows: list[dict] = []

    print("=" * 120)
    print("BENCHMARK STRATEGY x TIMEFRAME")
    print("=" * 120)
    print(f"Intervals: {list(ranges.keys())}")
    if common_start < common_end:
        print(f"Common window: {ms_to_iso(common_start)} -> {ms_to_iso(common_end)}")
    else:
        print("Common window invalide (pas de chevauchement).")
    print("-" * 120)

    for mode in modes:
        for interval, (raw_s, raw_e) in ranges.items():
            if mode == "common":
                if common_start >= common_end:
                    continue
                s_ms, e_ms = common_start, common_end
            else:
                s_ms, e_ms = clamp_range(raw_s, raw_e, user_start, user_end)
                if s_ms >= e_ms:
                    continue

            for name, overrides in strategies:
                base_args = make_base_args(
                    db=args.db,
                    interval=interval,
                    max_tokens=args.max_tokens,
                    min_candles=args.min_candles,
                )
                tuned = clone_args(base_args, overrides)
                res = run_portfolio_backtest(
                    tuned,
                    start_ms=s_ms,
                    end_ms=e_ms,
                    print_report=False,
                )
                if not res.get("ok"):
                    rows.append(
                        {
                            "mode": mode,
                            "interval": interval,
                            "strategy": name,
                            "start": ms_to_iso(s_ms),
                            "end": ms_to_iso(e_ms),
                            "ok": False,
                            "error": res.get("error", ""),
                        }
                    )
                    continue

                rows.append(
                    {
                        "mode": mode,
                        "interval": interval,
                        "strategy": name,
                        "start": ms_to_iso(s_ms),
                        "end": ms_to_iso(e_ms),
                        "ok": True,
                        "return_pct": float(res["return_pct"]),
                        "profit_factor": float(res["profit_factor"]) if res["profit_factor"] != float("inf") else 3.0,
                        "max_drawdown_pct": float(res["max_drawdown_pct"]),
                        "trades": int(res["total_trades"]),
                        "win_rate_pct": float(res["win_rate"]),
                        "final_equity": float(res["final_equity"]),
                    }
                )
                print(
                    f"{mode:6s} | {interval:4s} | {name:18s} | "
                    f"ret={res['return_pct']:+6.2f}% | pf={res['profit_factor']:.2f} | "
                    f"dd={res['max_drawdown_pct']:+6.2f}% | trades={int(res['total_trades']):4d}"
                )

    if not rows:
        print("Aucun resultat.")
        return 1

    df = pd.DataFrame(rows)
    ok_df = df[df["ok"] == True].copy()
    if ok_df.empty:
        print("Aucun test valide.")
        return 1

    print("-" * 120)
    for mode in modes:
        mdf = ok_df[ok_df["mode"] == mode].copy()
        if mdf.empty:
            continue
        print(f"Top results ({mode}):")
        top = mdf.sort_values(["return_pct", "profit_factor"], ascending=False).head(12)
        cols = [
            "interval",
            "strategy",
            "return_pct",
            "profit_factor",
            "max_drawdown_pct",
            "trades",
            "win_rate_pct",
            "start",
            "end",
        ]
        print(top[cols].to_string(index=False))
        print("-" * 120)

    out = Path(args.export)
    df.to_csv(out, index=False)
    print(f"Export: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
