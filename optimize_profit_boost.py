#!/usr/bin/env python3
"""
Recherche orientee performance pour pousser le rendement de la strategie.

Methode:
- Etape 1: random search sur WF 90/20/20 (rapide)
- Etape 2: validation des meilleurs sur WF 120/30/30 + full periode

Usage:
  python3 optimize_profit_boost.py --db squeeze_data.db --start 2025-07-19 --end 2026-02-12
"""

import argparse
import random
from pathlib import Path

import pandas as pd

from backtest_portfolio_hyperliquid import clone_args, parse_optional_date, run_portfolio_backtest


def wf_windows_90() -> list[tuple[str, str]]:
    return [
        ("2025-10-17", "2025-11-05"),
        ("2025-11-06", "2025-11-25"),
        ("2025-11-26", "2025-12-15"),
        ("2025-12-16", "2026-01-04"),
        ("2026-01-05", "2026-01-24"),
    ]


def wf_windows_120() -> list[tuple[str, str]]:
    return [
        ("2025-11-16", "2025-12-15"),
        ("2025-12-16", "2026-01-14"),
    ]


def compounded_return(returns_pct: list[float]) -> float:
    if not returns_pct:
        return 0.0
    c = 1.0
    for r in returns_pct:
        c *= 1.0 + (r / 100.0)
    return (c - 1.0) * 100.0


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
        stop_atr=1.70,
        target_atr=2.30,
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


def random_candidate(rng: random.Random) -> dict:
    pattern_min_rules = rng.choice([2, 3, 4])
    use_whitelist = rng.choice([False, True, True])  # biais leger pour tester la whitelist
    use_side_split = rng.choice([False, True])

    payload: dict[str, object] = {
        "min_squeeze_score": rng.choice([0.55, 0.60, 0.65]),
        "min_direction_confidence": rng.choice([0.45, 0.50, 0.55, 0.60]),
        "min_volume_ratio": rng.choice([0.20, 0.25, 0.30]),
        "stop_atr": rng.choice([1.4, 1.6, 1.8, 2.0]),
        "target_atr": rng.choice([2.0, 2.3, 2.6, 3.0]),
        "trailing_stop_pct": rng.choice([0.015, 0.020, 0.025]),
        "trailing_activation_pct": rng.choice([0.006, 0.008, 0.010]),
        "max_trades_per_day": rng.choice([4, 6, 8, 10]),
        "max_daily_loss_usd": rng.choice([8.0, 12.0, 15.0]),
        "max_holding_hours": rng.choice([12.0, 24.0, 36.0]),
        "enable_pattern_filter": True,
        "pattern_min_rules": pattern_min_rules,
        "pattern_rsi_max": rng.choice([44.0, 46.8, 50.0]),
        "pattern_ema_spread_max": rng.choice([-0.0025, -0.0018, -0.0013, -0.0008]),
        "pattern_ema_trend_slope_max": rng.choice([-0.0018, -0.0012, -0.0008, -0.0004]),
        "pattern_ret8_max": rng.choice([-0.006, -0.0035, -0.0023, 0.0]),
        "pattern_expected_move_max": rng.choice([0.070, 0.089, 0.110]),
        "enable_dynamic_whitelist": use_whitelist,
        "whitelist_lookback_days": rng.choice([20, 30, 45]),
        "whitelist_top_n": rng.choice([8, 12, 16]),
        "whitelist_min_trades": rng.choice([2, 3, 4]),
        "whitelist_score": rng.choice(["pnl", "expectancy"]),
    }

    if use_side_split:
        payload.update(
            {
                "min_squeeze_score_long": rng.choice([0.60, 0.65, 0.70]),
                "min_squeeze_score_short": rng.choice([0.50, 0.55, 0.60]),
                "min_direction_confidence_long": rng.choice([0.55, 0.60, 0.65]),
                "min_direction_confidence_short": rng.choice([0.40, 0.45, 0.50]),
                "min_volume_ratio_long": rng.choice([0.25, 0.30, 0.35]),
                "min_volume_ratio_short": rng.choice([0.15, 0.20, 0.25]),
                "min_expected_move_pct_long": rng.choice([0.020, 0.025, 0.030]),
                "min_expected_move_pct_short": rng.choice([0.010, 0.015, 0.020]),
                "stop_atr_long": rng.choice([1.6, 1.8, 2.0]),
                "stop_atr_short": rng.choice([1.2, 1.4, 1.6]),
                "target_atr_long": rng.choice([2.3, 2.6, 3.0]),
                "target_atr_short": rng.choice([1.8, 2.1, 2.3]),
                "trailing_stop_pct_long": rng.choice([0.018, 0.022, 0.025]),
                "trailing_stop_pct_short": rng.choice([0.012, 0.015, 0.018]),
                "trailing_activation_pct_long": rng.choice([0.008, 0.010, 0.012]),
                "trailing_activation_pct_short": rng.choice([0.004, 0.006, 0.008]),
            }
        )
    return payload


def evaluate_over_windows(args: argparse.Namespace, windows: list[tuple[str, str]]) -> dict:
    returns: list[float] = []
    pfs: list[float] = []
    dds: list[float] = []
    trades = 0
    ok = 0
    for s, e in windows:
        res = run_portfolio_backtest(
            args,
            start_ms=parse_optional_date(s),
            end_ms=parse_optional_date(e, end_of_day=True),
            print_report=False,
        )
        if not res.get("ok"):
            continue
        ok += 1
        r = float(res.get("return_pct", 0.0))
        pf = float(res.get("profit_factor", 0.0))
        dd = float(res.get("max_drawdown_pct", 0.0))
        returns.append(r)
        pfs.append(3.0 if pf == float("inf") else pf)
        dds.append(dd)
        trades += int(res.get("total_trades", 0))

    if ok == 0:
        return {
            "ok_windows": 0,
            "comp_ret_pct": -1e9,
            "mean_pf": 0.0,
            "mean_dd_pct": 0.0,
            "trades": 0,
            "pos_ratio": 0.0,
        }
    pos = sum(1 for r in returns if r > 0)
    return {
        "ok_windows": ok,
        "comp_ret_pct": compounded_return(returns),
        "mean_pf": sum(pfs) / len(pfs),
        "mean_dd_pct": sum(dds) / len(dds),
        "trades": trades,
        "pos_ratio": pos / len(returns),
    }


def stage1_score(row: dict) -> float:
    return (
        row["comp_ret_pct"]
        + 6.0 * (row["mean_pf"] - 1.0)
        + 4.0 * (row["pos_ratio"] - 0.5)
        - 0.2 * abs(row["mean_dd_pct"])
        + min(4.0, row["trades"] / 60.0)
    )


def stage2_score(row: dict) -> float:
    return (
        row["full_ret_pct"]
        + 0.35 * row["wf90_comp_ret_pct"]
        + 0.20 * row["wf120_comp_ret_pct"]
        + 8.0 * (row["full_pf"] - 1.0)
        + 4.0 * (row["wf90_pf"] - 1.0)
        - 0.30 * abs(row["full_dd_pct"])
    )


def run() -> int:
    parser = argparse.ArgumentParser(description="Profit-oriented optimizer for squeeze strategy")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--start", type=str, default="2025-07-19")
    parser.add_argument("--end", type=str, default="2026-02-12")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--stage2-top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-results", type=str, default="profit_boost_results.csv")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB introuvable: {db_path}")
        return 1

    rng = random.Random(args.seed)
    base_args = make_base_args(args.db, args.interval)
    wf90 = wf_windows_90()
    wf120 = wf_windows_120()

    seen = set()
    candidate_keys: set[str] = set()
    stage1_rows: list[dict] = []
    print("=" * 110)
    print("PROFIT BOOST SEARCH - STAGE 1 (WF 90/20/20)")
    print("=" * 110)
    print(f"Candidates: {args.candidates} | seed={args.seed}")
    print("-" * 110)
    idx = 0
    while idx < args.candidates:
        cand = random_candidate(rng)
        key = tuple(sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        candidate_keys.update(cand.keys())
        idx += 1
        tuned = clone_args(base_args, cand)
        m = evaluate_over_windows(tuned, wf90)
        score = stage1_score(m)
        row = {**cand, **m, "stage1_score": score}
        stage1_rows.append(row)
        print(
            f"[{idx:03d}/{args.candidates:03d}] "
            f"s1={score:+6.2f} | wf90={m['comp_ret_pct']:+6.2f}% pf={m['mean_pf']:.2f} "
            f"dd={m['mean_dd_pct']:+6.2f}% trades={m['trades']:4d}"
        )

    s1_df = pd.DataFrame(stage1_rows).sort_values("stage1_score", ascending=False).reset_index(drop=True)
    top_n = min(args.stage2_top, len(s1_df))
    top_df = s1_df.head(top_n).copy()

    print("-" * 110)
    print(f"STAGE 2 on top {top_n}")
    print("-" * 110)

    final_rows: list[dict] = []
    for i, (_, r) in enumerate(top_df.iterrows(), start=1):
        cand: dict[str, object] = {}
        for k in candidate_keys:
            if k not in r:
                continue
            v = r[k]
            if pd.isna(v):
                continue
            cand[k] = v
        tuned = clone_args(base_args, cand)
        m90 = evaluate_over_windows(tuned, wf90)
        m120 = evaluate_over_windows(tuned, wf120)
        full = run_portfolio_backtest(
            tuned,
            start_ms=parse_optional_date(args.start),
            end_ms=parse_optional_date(args.end, end_of_day=True),
            print_report=False,
        )
        full_ret = float(full.get("return_pct", 0.0)) if full.get("ok") else -1e9
        full_pf = float(full.get("profit_factor", 0.0)) if full.get("ok") else 0.0
        full_dd = float(full.get("max_drawdown_pct", 0.0)) if full.get("ok") else 0.0
        full_trades = int(full.get("total_trades", 0)) if full.get("ok") else 0

        row = {
            **cand,
            "wf90_comp_ret_pct": m90["comp_ret_pct"],
            "wf90_pf": m90["mean_pf"],
            "wf120_comp_ret_pct": m120["comp_ret_pct"],
            "wf120_pf": m120["mean_pf"],
            "full_ret_pct": full_ret,
            "full_pf": full_pf,
            "full_dd_pct": full_dd,
            "full_trades": full_trades,
        }
        row["stage2_score"] = stage2_score(row)
        final_rows.append(row)
        print(
            f"[{i:02d}/{top_n:02d}] s2={row['stage2_score']:+6.2f} | "
            f"full={full_ret:+6.2f}% pf={full_pf:.2f} dd={full_dd:+6.2f}% | "
            f"wf90={m90['comp_ret_pct']:+6.2f}% wf120={m120['comp_ret_pct']:+6.2f}%"
        )

    if not final_rows:
        print("Aucun resultat final.")
        return 1

    out_df = pd.DataFrame(final_rows).sort_values("stage2_score", ascending=False).reset_index(drop=True)
    print("-" * 110)
    show_cols = [
        "stage2_score",
        "full_ret_pct",
        "full_pf",
        "full_dd_pct",
        "wf90_comp_ret_pct",
        "wf120_comp_ret_pct",
        "min_squeeze_score",
        "min_direction_confidence",
        "stop_atr",
        "target_atr",
        "trailing_stop_pct",
        "trailing_activation_pct",
        "max_trades_per_day",
        "pattern_min_rules",
        "enable_dynamic_whitelist",
    ]
    print(out_df[show_cols].head(10).to_string(index=False))
    print("=" * 110)

    best = out_df.iloc[0]
    print("Best candidate command flags:")
    print(
        f"--min-squeeze-score {best['min_squeeze_score']:.2f} "
        f"--min-direction-confidence {best['min_direction_confidence']:.2f} "
        f"--stop-atr {best['stop_atr']:.2f} "
        f"--target-atr {best['target_atr']:.2f} "
        f"--trailing-stop-pct {best['trailing_stop_pct']:.3f} "
        f"--trailing-activation-pct {best['trailing_activation_pct']:.3f} "
        f"--max-trades-per-day {int(best['max_trades_per_day'])} "
        f"--enable-pattern-filter --pattern-min-rules {int(best['pattern_min_rules'])}"
    )
    if bool(best["enable_dynamic_whitelist"]):
        print(
            f"--enable-dynamic-whitelist "
            f"--whitelist-lookback-days {int(best['whitelist_lookback_days'])} "
            f"--whitelist-top-n {int(best['whitelist_top_n'])} "
            f"--whitelist-min-trades {int(best['whitelist_min_trades'])} "
            f"--whitelist-score {best['whitelist_score']}"
        )

    if args.export_results:
        out_df.to_csv(Path(args.export_results), index=False)
        print(f"\nExport: {args.export_results}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
