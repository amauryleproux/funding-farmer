#!/usr/bin/env python3
"""
Run des presets "prod" par timeframe sur toute la plage disponible.

Usage:
  python3 run_prod_timeframe_presets.py --db squeeze_data.db
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_portfolio_hyperliquid import clone_args, resolve_dataset_range, run_portfolio_backtest


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def make_base_args(db: str, interval: str) -> argparse.Namespace:
    return argparse.Namespace(
        db=db,
        interval=interval,
        start="",
        end="",
        max_tokens=25,
        min_candles=200,
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


def prod_presets() -> list[tuple[str, str, dict]]:
    # Choisis sur benchmark multi-timeframes deja lance.
    return [
        (
            "prod_15m_baseline",
            "15m",
            {
                "enable_pattern_filter": False,
                "min_squeeze_score": 0.60,
                "min_direction_confidence": 0.50,
                "min_volume_ratio": 0.25,
                "stop_atr": 1.7,
                "target_atr": 2.3,
                "trailing_stop_pct": 0.020,
                "trailing_activation_pct": 0.008,
                "max_trades_per_day": 6,
            },
        ),
        (
            "prod_30m_mean_revert",
            "30m",
            {
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
            "prod_1h_pattern_robust",
            "1h",
            {
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
    ]


def run() -> int:
    parser = argparse.ArgumentParser(description="Run production presets by timeframe")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--export", type=str, default="prod_timeframe_results.csv")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB introuvable: {db_path}")
        return 1

    rows: list[dict] = []
    print("=" * 110)
    print("PROD PRESETS BY TIMEFRAME (FULL AVAILABLE RANGE)")
    print("=" * 110)

    for name, interval, overrides in prod_presets():
        ds_start, ds_end = resolve_dataset_range(db_path, interval)
        if ds_start is None or ds_end is None:
            print(f"{name:24s} | {interval:4s} | pas de donnees")
            continue
        base = make_base_args(args.db, interval)
        tuned = clone_args(base, overrides)
        res = run_portfolio_backtest(
            tuned,
            start_ms=ds_start,
            end_ms=ds_end,
            print_report=False,
        )
        if not res.get("ok"):
            print(f"{name:24s} | {interval:4s} | ERROR: {res.get('error')}")
            rows.append(
                {
                    "preset": name,
                    "interval": interval,
                    "start": ms_to_iso(ds_start),
                    "end": ms_to_iso(ds_end),
                    "ok": False,
                    "error": res.get("error", ""),
                }
            )
            continue

        row = {
            "preset": name,
            "interval": interval,
            "start": ms_to_iso(ds_start),
            "end": ms_to_iso(ds_end),
            "ok": True,
            "return_pct": float(res["return_pct"]),
            "profit_factor": float(res["profit_factor"]) if res["profit_factor"] != float("inf") else 3.0,
            "max_drawdown_pct": float(res["max_drawdown_pct"]),
            "trades": int(res["total_trades"]),
            "win_rate_pct": float(res["win_rate"]),
            "final_equity": float(res["final_equity"]),
        }
        rows.append(row)
        print(
            f"{name:24s} | {interval:4s} | {row['start']} -> {row['end']} | "
            f"ret={row['return_pct']:+6.2f}% | pf={row['profit_factor']:.2f} | "
            f"dd={row['max_drawdown_pct']:+6.2f}% | trades={row['trades']:4d}"
        )

    if not rows:
        print("Aucun resultat.")
        return 1

    df = pd.DataFrame(rows)
    out = Path(args.export)
    df.to_csv(out, index=False)

    ok_df = df[df["ok"] == True].copy()
    if not ok_df.empty:
        print("-" * 110)
        print("Classement (par return_pct):")
        print(
            ok_df.sort_values(["return_pct", "profit_factor"], ascending=False)[
                [
                    "preset",
                    "interval",
                    "start",
                    "end",
                    "return_pct",
                    "profit_factor",
                    "max_drawdown_pct",
                    "trades",
                    "win_rate_pct",
                ]
            ].to_string(index=False)
        )
    print("=" * 110)
    print(f"Export: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

