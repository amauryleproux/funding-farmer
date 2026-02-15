#!/usr/bin/env python3
"""
Recherche du meilleur setup global de la strategie squeeze.

Methode:
- Evalue une grille de parametres
- Sur plusieurs schemas walk-forward (train/test/step)
- Score final base sur la robustesse OOS (retour compose, PF, constance, drawdown)

Usage:
  python3 optimize_best_setup.py --db squeeze_data.db
  python3 optimize_best_setup.py --db squeeze_data.db --max-candidates 60 --top-k 10
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest_portfolio_hyperliquid import (
    DAY_MS,
    clone_args,
    parse_optional_date,
    resolve_dataset_range,
    run_portfolio_backtest,
)


@dataclass
class WindowSpec:
    train_days: int
    test_days: int
    step_days: int


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


def parse_scheme_list(raw: str) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for block in (raw or "").split(","):
        part = block.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 3:
            continue
        try:
            tr, te, st = int(pieces[0]), int(pieces[1]), int(pieces[2])
        except ValueError:
            continue
        if tr > 0 and te > 0 and st > 0:
            specs.append(WindowSpec(tr, te, st))
    return specs


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


def build_candidates(args: argparse.Namespace) -> list[dict]:
    min_squeeze_scores = parse_float_list(args.grid_min_squeeze_score, [args.min_squeeze_score])
    min_direction_confs = parse_float_list(
        args.grid_min_direction_confidence, [args.min_direction_confidence]
    )
    stop_atrs = parse_float_list(args.grid_stop_atr, [args.stop_atr])
    target_atrs = parse_float_list(args.grid_target_atr, [args.target_atr])
    trailing_stops = parse_float_list(args.grid_trailing_stop_pct, [args.trailing_stop_pct])
    min_volume_ratios = parse_float_list(args.grid_min_volume_ratio, [args.min_volume_ratio])

    combos: list[dict] = []
    for mss, mdc, stop_atr, target_atr, trailing, mvr in product(
        min_squeeze_scores,
        min_direction_confs,
        stop_atrs,
        target_atrs,
        trailing_stops,
        min_volume_ratios,
    ):
        combos.append(
            {
                "min_squeeze_score": float(mss),
                "min_direction_confidence": float(mdc),
                "stop_atr": float(stop_atr),
                "target_atr": float(target_atr),
                "trailing_stop_pct": float(trailing),
                "min_volume_ratio": float(mvr),
            }
        )

    base = {
        "min_squeeze_score": float(args.min_squeeze_score),
        "min_direction_confidence": float(args.min_direction_confidence),
        "stop_atr": float(args.stop_atr),
        "target_atr": float(args.target_atr),
        "trailing_stop_pct": float(args.trailing_stop_pct),
        "min_volume_ratio": float(args.min_volume_ratio),
    }
    combos.insert(0, base)

    uniq: list[dict] = []
    seen = set()
    for c in combos:
        key = (
            round(c["min_squeeze_score"], 6),
            round(c["min_direction_confidence"], 6),
            round(c["stop_atr"], 6),
            round(c["target_atr"], 6),
            round(c["trailing_stop_pct"], 6),
            round(c["min_volume_ratio"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return cap_candidates(uniq, args.max_candidates)


def window_ranges(
    ds_start: int, ds_end: int, spec: WindowSpec, user_start: Optional[int], user_end: Optional[int]
) -> list[tuple[int, int, int, int]]:
    range_start = user_start if user_start is not None else ds_start
    range_end = user_end if user_end is not None else ds_end
    if range_start >= range_end:
        return []

    train_ms = spec.train_days * DAY_MS
    test_ms = spec.test_days * DAY_MS
    step_ms = spec.step_days * DAY_MS

    rows: list[tuple[int, int, int, int]] = []
    cursor = range_start
    while True:
        train_start = cursor
        train_end = train_start + train_ms - 1
        test_start = train_end + 1
        test_end = test_start + test_ms - 1
        if test_end > range_end:
            break
        rows.append((train_start, train_end, test_start, test_end))
        cursor += step_ms
    return rows


def profit_factor_from_trades(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_win = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum())
    gross_loss = float(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return abs(gross_win / gross_loss)


def evaluate_candidate(
    base_args: argparse.Namespace,
    candidate: dict,
    specs: list[WindowSpec],
    ds_start: int,
    ds_end: int,
    user_start: Optional[int],
    user_end: Optional[int],
) -> dict:
    tuned_args = clone_args(base_args, candidate)

    test_returns: list[float] = []
    test_dds: list[float] = []
    test_windows = 0
    positive_windows = 0
    all_test_trades: list[pd.DataFrame] = []

    for spec in specs:
        # Ajuste min_candles a la taille des fenetres pour eviter zero token eligibles.
        max_train_bars = spec.train_days * 24
        max_test_bars = spec.test_days * 24
        effective_min_candles = int(
            min(
                tuned_args.min_candles,
                max(1, int(max_train_bars * 0.90)),
                max(1, int(max_test_bars * 0.90)),
            )
        )
        effective_min_candles = max(tuned_args.warmup_bars + 1, effective_min_candles)
        spec_args = clone_args(tuned_args, {"min_candles": effective_min_candles})

        windows = window_ranges(ds_start, ds_end, spec, user_start, user_end)
        for train_start, train_end, test_start, test_end in windows:
            train_res = run_portfolio_backtest(
                spec_args,
                start_ms=train_start,
                end_ms=train_end,
                print_report=False,
            )
            test_res = run_portfolio_backtest(
                spec_args,
                start_ms=test_start,
                end_ms=test_end,
                print_report=False,
            )

            if not train_res.get("ok") or not test_res.get("ok"):
                continue

            tr = float(test_res.get("return_pct", 0.0))
            dd = float(test_res.get("max_drawdown_pct", 0.0))
            test_returns.append(tr)
            test_dds.append(dd)
            test_windows += 1
            if tr > 0:
                positive_windows += 1

            trades_df = test_res.get("trades_df")
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                all_test_trades.append(trades_df)

    if test_windows == 0:
        return {
            **candidate,
            "score": -1e12,
            "windows": 0,
            "test_return_compounded_pct": 0.0,
            "test_return_mean_pct": 0.0,
            "test_return_median_pct": 0.0,
            "test_return_std_pct": 0.0,
            "test_worst_window_pct": 0.0,
            "test_positive_windows_pct": 0.0,
            "test_total_trades": 0,
            "test_win_rate_pct": 0.0,
            "test_profit_factor": 0.0,
            "test_avg_max_dd_pct": 0.0,
        }

    compounded = 1.0
    for r in test_returns:
        compounded *= (1.0 + r / 100.0)
    compounded_pct = (compounded - 1.0) * 100.0
    mean_ret = float(np.mean(test_returns))
    median_ret = float(np.median(test_returns))
    std_ret = float(np.std(test_returns))
    worst_ret = float(np.min(test_returns))
    pos_ratio = positive_windows / test_windows
    avg_dd = float(np.mean(test_dds)) if test_dds else 0.0

    trades = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    total_trades = int(len(trades))
    win_rate = float((trades["net_pnl"] > 0).mean() * 100) if total_trades else 0.0
    pf = profit_factor_from_trades(trades)
    if math.isinf(pf):
        pf = 3.0

    # Score robuste (penalise instabilite et drawdown)
    score = (
        compounded_pct
        + 12.0 * (pf - 1.0)
        + 8.0 * (pos_ratio - 0.5)
        - 0.35 * avg_dd
        - 0.45 * std_ret
        + min(0.0, worst_ret) * 0.35
        + min(8.0, total_trades / 40.0)
    )

    return {
        **candidate,
        "score": float(score),
        "windows": int(test_windows),
        "test_return_compounded_pct": float(compounded_pct),
        "test_return_mean_pct": mean_ret,
        "test_return_median_pct": median_ret,
        "test_return_std_pct": std_ret,
        "test_worst_window_pct": worst_ret,
        "test_positive_windows_pct": pos_ratio * 100.0,
        "test_total_trades": total_trades,
        "test_win_rate_pct": win_rate,
        "test_profit_factor": float(pf),
        "test_avg_max_dd_pct": avg_dd,
    }


def evaluate_candidate_job(payload: tuple[dict, dict, list[tuple[int, int, int]], int, int, Optional[int], Optional[int]]) -> dict:
    base_args_dict, candidate, specs_tuples, ds_start, ds_end, user_start, user_end = payload
    base_args = argparse.Namespace(**base_args_dict)
    specs = [WindowSpec(train_days=t[0], test_days=t[1], step_days=t[2]) for t in specs_tuples]
    return evaluate_candidate(
        base_args=base_args,
        candidate=candidate,
        specs=specs,
        ds_start=ds_start,
        ds_end=ds_end,
        user_start=user_start,
        user_end=user_end,
    )


def print_progress(idx: int, total: int, res: dict) -> None:
    print(
        f"[{idx:03d}/{total:03d}] "
        f"score={res['score']:+7.2f} | comp={res['test_return_compounded_pct']:+6.2f}% | "
        f"pf={res['test_profit_factor']:.2f} | windows={res['windows']:2d} | "
        f"mss={res['min_squeeze_score']:.2f} conf={res['min_direction_confidence']:.2f} "
        f"sl={res['stop_atr']:.2f} tp={res['target_atr']:.2f} tr={res['trailing_stop_pct']:.3f} vr={res['min_volume_ratio']:.2f}"
    )


def run() -> int:
    parser = argparse.ArgumentParser(description="Global setup optimizer across walk-forward schemes")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--wf-schemes", type=str, default="120:30:30,90:20:20")
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)

    # backtest params (same semantics as backtest_portfolio_hyperliquid.py)
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

    # grid
    parser.add_argument("--grid-min-squeeze-score", type=str, default="0.50,0.55,0.60")
    parser.add_argument("--grid-min-direction-confidence", type=str, default="0.55,0.60,0.65")
    parser.add_argument("--grid-stop-atr", type=str, default="1.3,1.5,1.7")
    parser.add_argument("--grid-target-atr", type=str, default="2.5,3.0,3.5")
    parser.add_argument("--grid-trailing-stop-pct", type=str, default="0.012,0.015,0.020")
    parser.add_argument("--grid-min-volume-ratio", type=str, default="0.25,0.30,0.35")

    parser.add_argument("--export-results", type=str, default="")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB introuvable: {db_path}")
        return 1

    schemes = parse_scheme_list(args.wf_schemes)
    if not schemes:
        print("Aucun schema walk-forward valide (--wf-schemes).")
        return 1

    ds_start, ds_end = resolve_dataset_range(db_path)
    if ds_start is None or ds_end is None:
        print("Impossible de lire la plage de donnees hyperliquid.")
        return 1

    user_start = parse_optional_date(args.start)
    user_end = parse_optional_date(args.end, end_of_day=True)
    if user_start and user_end and user_start > user_end:
        print("Erreur: --start doit etre <= --end")
        return 1

    candidates = build_candidates(args)
    print("=" * 110)
    print("GLOBAL SETUP SEARCH")
    print("=" * 110)
    print(
        f"Range: {args.start or 'debut'} -> {args.end or 'fin'} | "
        f"Schemes={args.wf_schemes} | Candidats={len(candidates)}"
    )
    print("-" * 110)

    rows: list[dict] = []
    total = len(candidates)
    if args.workers <= 1:
        for idx, cand in enumerate(candidates, start=1):
            res = evaluate_candidate(
                base_args=args,
                candidate=cand,
                specs=schemes,
                ds_start=ds_start,
                ds_end=ds_end,
                user_start=user_start,
                user_end=user_end,
            )
            rows.append(res)
            print_progress(idx, total, res)
    else:
        workers = max(1, int(args.workers))
        specs_tuples = [(s.train_days, s.test_days, s.step_days) for s in schemes]
        base_args_dict = vars(args).copy()

        def run_parallel(executor_cls) -> None:
            with executor_cls(max_workers=workers) as ex:
                futures = {}
                for cand in candidates:
                    payload = (base_args_dict, cand, specs_tuples, ds_start, ds_end, user_start, user_end)
                    futures[ex.submit(evaluate_candidate_job, payload)] = cand

                done = 0
                for fut in as_completed(futures):
                    cand = futures[fut]
                    done += 1
                    try:
                        res = fut.result()
                    except Exception:
                        res = {
                            **cand,
                            "score": -1e12,
                            "windows": 0,
                            "test_return_compounded_pct": 0.0,
                            "test_return_mean_pct": 0.0,
                            "test_return_median_pct": 0.0,
                            "test_return_std_pct": 0.0,
                            "test_worst_window_pct": 0.0,
                            "test_positive_windows_pct": 0.0,
                            "test_total_trades": 0,
                            "test_win_rate_pct": 0.0,
                            "test_profit_factor": 0.0,
                            "test_avg_max_dd_pct": 0.0,
                        }
                    rows.append(res)
                    print_progress(done, total, res)

        try:
            run_parallel(ProcessPoolExecutor)
        except (PermissionError, OSError):
            print("Parallel process workers unavailable in this environment, fallback to threads.")
            run_parallel(ThreadPoolExecutor)

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if df.empty:
        print("Aucun resultat.")
        return 1

    print("-" * 110)
    print(f"Top {min(args.top_k, len(df))} setups:")
    cols = [
        "score",
        "test_return_compounded_pct",
        "test_return_mean_pct",
        "test_return_std_pct",
        "test_worst_window_pct",
        "test_positive_windows_pct",
        "test_profit_factor",
        "test_total_trades",
        "test_avg_max_dd_pct",
        "min_squeeze_score",
        "min_direction_confidence",
        "stop_atr",
        "target_atr",
        "trailing_stop_pct",
        "min_volume_ratio",
    ]
    print(df[cols].head(args.top_k).to_string(index=False))
    print("=" * 110)

    best = df.iloc[0]
    print("Setup recommande (best score):")
    print(
        f"--min-squeeze-score {best['min_squeeze_score']:.2f} "
        f"--min-direction-confidence {best['min_direction_confidence']:.2f} "
        f"--stop-atr {best['stop_atr']:.2f} "
        f"--target-atr {best['target_atr']:.2f} "
        f"--trailing-stop-pct {best['trailing_stop_pct']:.3f} "
        f"--min-volume-ratio {best['min_volume_ratio']:.2f}"
    )

    if args.export_results:
        out = Path(args.export_results)
        df.to_csv(out, index=False)
        print(f"\nExport: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
