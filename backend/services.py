from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_portfolio_hyperliquid import (
    clone_args,
    ms_to_iso,
    parse_optional_date,
    resolve_dataset_range,
    run_portfolio_backtest,
)
from benchmark_timeframes_strategies import make_base_args, strategy_presets


ALLOWED_TIMEFRAMES = ("15m", "30m", "1h")
DEFAULT_DB = "squeeze_data.db"
AI_CACHE_DIR = Path(".cache/ai_backtests")

AI_MUTABLE_RULES: dict[str, tuple[float, float, str]] = {
    "min_squeeze_score": (0.35, 0.90, "float"),
    "min_direction_confidence": (0.30, 0.95, "float"),
    "min_volume_ratio": (0.05, 2.50, "float"),
    "min_expected_move_pct": (0.005, 0.20, "float"),
    "min_entry_score": (0.30, 0.95, "float"),
    "stop_atr": (0.8, 4.0, "float"),
    "target_atr": (1.0, 6.0, "float"),
    "trailing_stop_pct": (0.004, 0.08, "float"),
    "trailing_activation_pct": (0.002, 0.05, "float"),
    "max_holding_hours": (2.0, 120.0, "float"),
    "max_trades_per_day": (1.0, 20.0, "int"),
    "breakout_min_vol_ratio": (0.6, 3.0, "float"),
    "ema_cross_min_trend_slope": (0.0001, 0.005, "float"),
    "rsi_revert_long_rsi": (20.0, 45.0, "float"),
    "rsi_revert_short_rsi": (55.0, 80.0, "float"),
    "rsi_revert_long_bb_pos_max": (0.05, 0.40, "float"),
    "rsi_revert_short_bb_pos_min": (0.60, 0.95, "float"),
    "rsi_revert_max_adverse_trend_slope": (0.0004, 0.004, "float"),
    "macd_min_hist_pct": (0.00005, 0.0035, "float"),
    "macd_min_trend_slope": (0.0001, 0.004, "float"),
}


STRATEGY_DETAILS: dict[str, dict[str, str]] = {
    "baseline": {
        "name": "Baseline Squeeze",
        "description": "Squeeze breakout classique, filtres équilibrés.",
    },
    "pattern_robust": {
        "name": "Pattern Robust",
        "description": "Squeeze avec pattern filter strict orienté robustesse.",
    },
    "pattern_aggressive": {
        "name": "Pattern Aggressive",
        "description": "Squeeze pattern plus agressif pour augmenter la fréquence.",
    },
    "trend_follow": {
        "name": "Trend Follow",
        "description": "Suivi de tendance avec filtres de momentum renforcés.",
    },
    "mean_revert_squeeze": {
        "name": "Mean Revert Squeeze",
        "description": "Squeeze orienté retour à la moyenne.",
    },
    "split_long_short": {
        "name": "Split Long/Short",
        "description": "Paramètres asymétriques long et short.",
    },
    "breakout_momentum": {
        "name": "Breakout Momentum",
        "description": "Breakout directionnel avec confirmation volume.",
    },
    "ema_cross_trend": {
        "name": "EMA Cross Trend",
        "description": "Croisement EMA avec filtre de tendance.",
    },
    "rsi_reversion_classic": {
        "name": "RSI Reversion Classic",
        "description": "Réversion RSI + Bollinger bands.",
    },
    "macd_trend_confirmed": {
        "name": "MACD Trend Confirmed",
        "description": "Croisement MACD avec confirmations tendance et zéro line.",
    },
}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _clean_records(df: pd.DataFrame, limit: int, offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
    if df.empty:
        return 0, []
    safe_limit = max(1, min(5000, int(limit)))
    safe_offset = max(0, int(offset))
    total = int(len(df))
    sliced = df.iloc[safe_offset : safe_offset + safe_limit].copy()
    sliced = sliced.where(pd.notna(sliced), None)
    rows = []
    for rec in sliced.to_dict("records"):
        rows.append({k: _normalize_value(v) for k, v in rec.items()})
    return total, rows


def _strategy_map() -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for strategy_id, overrides in strategy_presets():
        details = STRATEGY_DETAILS.get(strategy_id, {"name": strategy_id, "description": ""})
        mapped[strategy_id] = {
            "id": strategy_id,
            "name": details["name"],
            "description": details["description"],
            "entry_mode": str(overrides.get("entry_mode", "squeeze")),
            "defaults": dict(overrides),
        }
    return mapped


def get_strategy_catalog() -> list[dict[str, Any]]:
    rows = list(_strategy_map().values())
    rows.sort(key=lambda x: x["id"])
    return rows


def get_strategy(strategy_id: str) -> dict[str, Any]:
    mapped = _strategy_map()
    strategy = mapped.get(strategy_id)
    if not strategy:
        raise ValueError(f"Strategie inconnue: {strategy_id}")
    return strategy


def _clamp_range(raw_start: int, raw_end: int, user_start: int | None, user_end: int | None) -> tuple[int, int]:
    start_ms = max(raw_start, user_start) if user_start is not None else raw_start
    end_ms = min(raw_end, user_end) if user_end is not None else raw_end
    return start_ms, end_ms


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_cache_key(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


def _read_ai_cache(cache_key: str) -> dict[str, Any] | None:
    cache_path = AI_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def _write_ai_cache(cache_key: str, result: dict[str, Any]) -> None:
    AI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = AI_CACHE_DIR / f"{cache_key}.json"
    cache_path.write_text(_canonical_json(result))


def _build_summary(
    *,
    strategy: dict[str, Any],
    timeframe: str,
    db: str,
    args_payload: dict[str, Any],
    start_ms: int | None,
    end_ms: int | None,
    result: dict[str, Any],
    trades_df: pd.DataFrame,
) -> dict[str, Any]:
    by_reason: list[dict[str, Any]] = []
    if not trades_df.empty:
        grp = trades_df.groupby("reason")["net_pnl"].agg(["count", "sum"]).reset_index()
        grp = grp.sort_values("sum", ascending=False)
        for row in grp.to_dict("records"):
            by_reason.append(
                {
                    "reason": str(row["reason"]),
                    "count": int(row["count"]),
                    "pnl": float(row["sum"]),
                }
            )

    return {
        "strategy_id": strategy["id"],
        "strategy_name": strategy["name"],
        "entry_mode": str(args_payload.get("entry_mode", strategy["entry_mode"])),
        "timeframe": timeframe,
        "db": db,
        "window": {
            "start": ms_to_iso(start_ms) if start_ms else None,
            "end": ms_to_iso(end_ms) if end_ms else None,
        },
        "performance": {
            "final_equity": float(result["final_equity"]),
            "total_pnl": float(result["total_pnl"]),
            "return_pct": float(result["return_pct"]),
            "win_rate_pct": float(result["win_rate"]),
            "profit_factor": 3.0 if result["profit_factor"] == float("inf") else float(result["profit_factor"]),
            "max_drawdown_pct": float(result["max_drawdown_pct"]),
            "trades": int(result["total_trades"]),
        },
        "universe": {
            "selected_symbols": int(result["selected_symbols"]),
            "tested_symbols": int(result["tested_symbols"]),
        },
        "execution": {
            "filter_stats": result.get("filter_stats", {}),
            "trades_by_reason": by_reason,
        },
    }


def run_single_backtest(payload: dict[str, Any], report: Any | None = None) -> dict[str, Any]:
    emit = report or (lambda _p, _m: None)

    timeframe = str(payload.get("timeframe", "1h"))
    if timeframe not in ALLOWED_TIMEFRAMES:
        raise ValueError(f"timeframe invalide: {timeframe}")

    db = str(payload.get("db", DEFAULT_DB))
    strategy_id = str(payload.get("strategy_id", "")).strip()
    strategy = get_strategy(strategy_id)

    emit(0.05, "building_args")
    max_tokens = int(payload.get("max_tokens", 25))
    min_candles = int(payload.get("min_candles", 200))
    base_args = make_base_args(db=db, interval=timeframe, max_tokens=max_tokens, min_candles=min_candles)

    merged_overrides: dict[str, Any] = dict(strategy["defaults"])
    user_overrides = payload.get("overrides", {}) or {}
    if not isinstance(user_overrides, dict):
        raise ValueError("overrides doit etre un objet JSON (dict)")
    merged_overrides.update(user_overrides)

    tuned_args = clone_args(base_args, merged_overrides)
    start_ms = parse_optional_date(str(payload.get("start", "") or ""))
    end_ms = parse_optional_date(str(payload.get("end", "") or ""), end_of_day=True)
    if start_ms and end_ms and start_ms > end_ms:
        raise ValueError("--start doit etre <= --end")

    emit(0.20, "running_backtest")
    result = run_portfolio_backtest(
        tuned_args,
        start_ms=start_ms,
        end_ms=end_ms,
        print_report=False,
    )
    if not result.get("ok"):
        raise ValueError(str(result.get("error", "Erreur backtest")))

    trades_df = result["trades_df"].copy()
    equity_df = result["equity_df"].copy()
    summary = _build_summary(
        strategy=strategy,
        timeframe=timeframe,
        db=db,
        args_payload=vars(tuned_args),
        start_ms=start_ms,
        end_ms=end_ms,
        result=result,
        trades_df=trades_df,
    )
    emit(1.0, "completed")
    return {
        "kind": "single",
        "summary": summary,
        "runtime_args": vars(tuned_args),
        "trades_df": trades_df,
        "equity_df": equity_df,
    }


def run_compare_backtests(payload: dict[str, Any], report: Any | None = None) -> dict[str, Any]:
    emit = report or (lambda _p, _m: None)

    db = str(payload.get("db", DEFAULT_DB))
    strategy_map = _strategy_map()
    requested_ids = payload.get("strategy_ids", [])
    if requested_ids:
        strategy_ids = [str(x) for x in requested_ids]
    else:
        strategy_ids = sorted(strategy_map.keys())

    for strategy_id in strategy_ids:
        if strategy_id not in strategy_map:
            raise ValueError(f"Strategie inconnue: {strategy_id}")

    requested_timeframes = payload.get("timeframes", list(ALLOWED_TIMEFRAMES))
    timeframes = [str(x) for x in requested_timeframes]
    if not timeframes:
        raise ValueError("timeframes vide")
    for timeframe in timeframes:
        if timeframe not in ALLOWED_TIMEFRAMES:
            raise ValueError(f"timeframe invalide: {timeframe}")

    window_mode = str(payload.get("window_mode", "both"))
    if window_mode not in {"common", "full", "both"}:
        raise ValueError("window_mode doit etre common, full ou both")

    user_start = parse_optional_date(str(payload.get("start", "") or ""))
    user_end = parse_optional_date(str(payload.get("end", "") or ""), end_of_day=True)
    if user_start and user_end and user_start > user_end:
        raise ValueError("--start doit etre <= --end")

    ranges: dict[str, tuple[int, int]] = {}
    for timeframe in timeframes:
        start_ms, end_ms = resolve_dataset_range(Path(db), timeframe)
        if start_ms is None or end_ms is None:
            continue
        ranges[timeframe] = (start_ms, end_ms)
    if not ranges:
        raise ValueError("Aucune plage de donnees valide pour ces timeframes")

    common_start = max(v[0] for v in ranges.values())
    common_end = min(v[1] for v in ranges.values())
    if user_start is not None:
        common_start = max(common_start, user_start)
    if user_end is not None:
        common_end = min(common_end, user_end)

    max_tokens = int(payload.get("max_tokens", 25))
    min_candles = int(payload.get("min_candles", 200))
    shared_overrides = payload.get("overrides", {}) or {}
    if not isinstance(shared_overrides, dict):
        raise ValueError("overrides doit etre un objet JSON (dict)")

    modes = ["common", "full"] if window_mode == "both" else [window_mode]
    rows: list[dict[str, Any]] = []
    total_runs = len(modes) * len(ranges) * len(strategy_ids)
    done = 0

    for mode in modes:
        for timeframe, (raw_start, raw_end) in ranges.items():
            if mode == "common":
                if common_start >= common_end:
                    continue
                start_ms, end_ms = common_start, common_end
            else:
                start_ms, end_ms = _clamp_range(raw_start, raw_end, user_start, user_end)
                if start_ms >= end_ms:
                    continue

            for strategy_id in strategy_ids:
                strategy = strategy_map[strategy_id]
                base_args = make_base_args(
                    db=db,
                    interval=timeframe,
                    max_tokens=max_tokens,
                    min_candles=min_candles,
                )
                overrides = dict(strategy["defaults"])
                overrides.update(shared_overrides)
                tuned_args = clone_args(base_args, overrides)
                res = run_portfolio_backtest(
                    tuned_args,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    print_report=False,
                )
                row: dict[str, Any] = {
                    "mode": mode,
                    "strategy_id": strategy_id,
                    "strategy_name": strategy["name"],
                    "timeframe": timeframe,
                    "start": ms_to_iso(start_ms),
                    "end": ms_to_iso(end_ms),
                    "ok": bool(res.get("ok")),
                }
                if res.get("ok"):
                    row.update(
                        {
                            "return_pct": float(res["return_pct"]),
                            "profit_factor": (
                                3.0 if res["profit_factor"] == float("inf") else float(res["profit_factor"])
                            ),
                            "max_drawdown_pct": float(res["max_drawdown_pct"]),
                            "trades": int(res["total_trades"]),
                            "win_rate_pct": float(res["win_rate"]),
                            "final_equity": float(res["final_equity"]),
                        }
                    )
                else:
                    row["error"] = str(res.get("error", "Erreur backtest"))
                rows.append(row)
                done += 1
                emit(min(0.98, done / max(1, total_runs)), f"running_{done}_{total_runs}")

    leaderboard = [r for r in rows if r.get("ok")]
    leaderboard.sort(key=lambda x: (x.get("return_pct", -1e9), x.get("profit_factor", -1e9)), reverse=True)
    emit(1.0, "completed")
    return {
        "kind": "compare",
        "rows": rows,
        "leaderboard": leaderboard,
        "summary": {
            "runs_total": len(rows),
            "runs_ok": len(leaderboard),
            "strategies_tested": strategy_ids,
            "timeframes_tested": list(ranges.keys()),
            "window_mode": window_mode,
        },
    }


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _clamp_numeric(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def _mutate_overrides(
    base_args: Any,
    base_overrides: dict[str, Any],
    rng: random.Random,
    force_mutation: bool,
) -> dict[str, Any]:
    out = dict(base_overrides)
    keys = list(AI_MUTABLE_RULES.keys())
    rng.shuffle(keys)
    mutation_count = rng.randint(1, 6) if force_mutation else 0

    for key in keys[:mutation_count]:
        rule = AI_MUTABLE_RULES[key]
        min_v, max_v, value_type = rule
        current = out.get(key, getattr(base_args, key, None))
        if current is None:
            continue
        current_f = _safe_float(current, (min_v + max_v) / 2.0)
        if value_type == "int":
            delta = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            mutated = _clamp_numeric(current_f + delta, min_v, max_v)
            out[key] = int(round(mutated))
            continue
        stretch = rng.uniform(-0.22, 0.22)
        base = current_f * (1.0 + stretch)
        if abs(current_f) < 1e-9:
            base = rng.uniform(min_v, max_v)
        mutated = _clamp_numeric(base, min_v, max_v)
        out[key] = round(mutated, 6)

    # Toggle selectively for booleans that matter for entry modes.
    for bool_key in (
        "breakout_require_trend",
        "ema_cross_require_trend",
        "macd_require_zero_line",
        "macd_require_trend",
        "rsi_revert_trend_filter",
        "enable_pattern_filter",
    ):
        if bool_key not in out:
            continue
        if rng.random() < 0.20 and force_mutation:
            out[bool_key] = not bool(out[bool_key])

    return out


def _score_ai_candidate(row: dict[str, Any], objective: str) -> float:
    ret = _safe_float(row.get("return_pct"), 0.0)
    pf = _safe_float(row.get("profit_factor"), 0.0)
    dd = abs(_safe_float(row.get("max_drawdown_pct"), 0.0))
    wr = _safe_float(row.get("win_rate_pct"), 0.0)
    trades = _safe_float(row.get("trades"), 0.0)

    if objective == "return_only":
        return ret
    if objective == "low_drawdown":
        return ret - 0.70 * dd + (pf - 1.0) * 1.4
    if objective == "consistency":
        return ret + (pf - 1.0) * 2.2 + (wr / 100.0) * 2.0 - 0.35 * dd + min(2.0, trades / 40.0)
    return ret + (pf - 1.0) * 1.8 - 0.45 * dd + (wr / 100.0)


def _window_bounds_for_timeframe(
    *,
    db: str,
    timeframe: str,
    user_start: int | None,
    user_end: int | None,
) -> tuple[int, int] | None:
    range_start, range_end = resolve_dataset_range(Path(db), timeframe)
    if range_start is None or range_end is None:
        return None
    start_ms, end_ms = _clamp_range(range_start, range_end, user_start, user_end)
    if start_ms >= end_ms:
        return None
    return start_ms, end_ms


def run_ai_backtests(payload: dict[str, Any], report: Any | None = None) -> dict[str, Any]:
    emit = report or (lambda _p, _m: None)

    db = str(payload.get("db", DEFAULT_DB))
    strategy_map = _strategy_map()
    requested_ids = payload.get("strategy_ids", [])
    strategy_ids = [str(x) for x in requested_ids] if requested_ids else sorted(strategy_map.keys())
    for strategy_id in strategy_ids:
        if strategy_id not in strategy_map:
            raise ValueError(f"Strategie inconnue: {strategy_id}")

    requested_timeframes = payload.get("timeframes", list(ALLOWED_TIMEFRAMES))
    timeframes = [str(x) for x in requested_timeframes if str(x) in ALLOWED_TIMEFRAMES]
    if not timeframes:
        raise ValueError("Aucun timeframe valide")

    user_start = parse_optional_date(str(payload.get("start", "") or ""))
    user_end = parse_optional_date(str(payload.get("end", "") or ""), end_of_day=True)
    if user_start and user_end and user_start > user_end:
        raise ValueError("--start doit etre <= --end")

    max_tokens = _safe_int(payload.get("max_tokens", 25), 25)
    min_candles = _safe_int(payload.get("min_candles", 200), 200)
    max_runs = max(10, min(2000, _safe_int(payload.get("max_runs", 120), 120)))
    top_n = max(1, min(20, _safe_int(payload.get("top_n", 5), 5)))
    min_trades = max(0, _safe_int(payload.get("min_trades", 5), 5))
    max_drawdown_abs_pct = max(0.1, _safe_float(payload.get("max_drawdown_pct", 25.0), 25.0))
    objective = str(payload.get("objective", "balanced"))
    if objective not in {"balanced", "return_only", "low_drawdown", "consistency"}:
        objective = "balanced"

    shared_overrides = payload.get("overrides", {}) or {}
    if not isinstance(shared_overrides, dict):
        raise ValueError("overrides doit etre un objet JSON (dict)")

    seed = _safe_int(payload.get("seed", 42), 42)
    force_refresh = bool(payload.get("force_refresh", False))
    rng = random.Random(seed)

    pairs: list[tuple[str, str, int, int]] = []
    for timeframe in timeframes:
        bounds = _window_bounds_for_timeframe(
            db=db,
            timeframe=timeframe,
            user_start=user_start,
            user_end=user_end,
        )
        if not bounds:
            continue
        for strategy_id in strategy_ids:
            pairs.append((strategy_id, timeframe, bounds[0], bounds[1]))
    if not pairs:
        raise ValueError("Aucune combinaison strategie/timeframe exploitable")

    db_path = Path(db)
    db_mtime_ns = db_path.stat().st_mtime_ns if db_path.exists() else -1
    cache_signature = {
        "db": str(db_path.resolve()) if db_path.exists() else db,
        "db_mtime_ns": db_mtime_ns,
        "strategy_ids": sorted(strategy_ids),
        "timeframes": sorted(timeframes),
        "window_pairs": [
            {"strategy_id": s, "timeframe": tf, "start_ms": st, "end_ms": en}
            for s, tf, st, en in sorted(pairs, key=lambda x: (x[0], x[1], x[2], x[3]))
        ],
        "max_tokens": max_tokens,
        "min_candles": min_candles,
        "max_runs": max_runs,
        "top_n": top_n,
        "min_trades": min_trades,
        "max_drawdown_pct": round(max_drawdown_abs_pct, 6),
        "objective": objective,
        "seed": seed,
        "shared_overrides": shared_overrides,
    }
    cache_key = _payload_cache_key(cache_signature)
    if not force_refresh:
        cached = _read_ai_cache(cache_key)
        if cached:
            summary = dict(cached.get("summary", {}))
            summary["cached"] = True
            summary["cache_key"] = cache_key
            emit(1.0, "completed_cached")
            return {
                "kind": "ai_optimize",
                "top_picks": cached.get("top_picks", []),
                "leaderboard": cached.get("top_picks", []),
                "explored": cached.get("explored", []),
                "summary": summary,
            }

    runs_per_pair = max(1, max_runs // len(pairs))
    total_planned_runs = runs_per_pair * len(pairs)
    if total_planned_runs < max_runs:
        total_planned_runs += min(max_runs - total_planned_runs, len(pairs))

    explored: list[dict[str, Any]] = []
    done = 0
    for pair_idx, (strategy_id, timeframe, start_ms, end_ms) in enumerate(pairs):
        strategy = strategy_map[strategy_id]
        base_args = make_base_args(
            db=db,
            interval=timeframe,
            max_tokens=max_tokens,
            min_candles=min_candles,
        )
        base_overrides = dict(strategy["defaults"])
        base_overrides.update(shared_overrides)

        local_runs = runs_per_pair + (1 if pair_idx < (total_planned_runs - runs_per_pair * len(pairs)) else 0)
        for run_idx in range(local_runs):
            use_base = run_idx == 0
            overrides = _mutate_overrides(
                base_args=base_args,
                base_overrides=base_overrides,
                rng=rng,
                force_mutation=not use_base,
            )
            tuned_args = clone_args(base_args, overrides)
            res = run_portfolio_backtest(
                tuned_args,
                start_ms=start_ms,
                end_ms=end_ms,
                print_report=False,
            )

            row: dict[str, Any] = {
                "strategy_id": strategy_id,
                "strategy_name": strategy["name"],
                "timeframe": timeframe,
                "start": ms_to_iso(start_ms),
                "end": ms_to_iso(end_ms),
                "run_id": f"{strategy_id}:{timeframe}:{run_idx}",
                "seed": seed,
                "is_base": use_base,
                "ok": bool(res.get("ok")),
            }
            if res.get("ok"):
                pf = 3.0 if res["profit_factor"] == float("inf") else float(res["profit_factor"])
                dd = float(res["max_drawdown_pct"])
                trades = int(res["total_trades"])
                row.update(
                    {
                        "return_pct": float(res["return_pct"]),
                        "profit_factor": pf,
                        "max_drawdown_pct": dd,
                        "trades": trades,
                        "win_rate_pct": float(res["win_rate"]),
                        "final_equity": float(res["final_equity"]),
                        "overrides": overrides,
                    }
                )
                quality_ok = (
                    row["return_pct"] > 0.0
                    and pf >= 1.0
                    and trades >= min_trades
                    and abs(dd) <= max_drawdown_abs_pct
                )
                row["quality_ok"] = quality_ok
                row["ai_score"] = _score_ai_candidate(row, objective)
            else:
                row["error"] = str(res.get("error", "Erreur backtest"))
                row["quality_ok"] = False
                row["ai_score"] = -1e12
            explored.append(row)
            done += 1
            emit(min(0.98, done / max(1, total_planned_runs)), f"ai_run_{done}_{total_planned_runs}")

    ok_rows = [r for r in explored if r.get("ok")]
    quality_rows = [r for r in ok_rows if r.get("quality_ok")]
    ranked_pool = quality_rows if quality_rows else ok_rows
    ranked_pool.sort(key=lambda r: (r.get("ai_score", -1e12), r.get("return_pct", -1e12)), reverse=True)
    top_picks = ranked_pool[:top_n]

    # Compact output: avoid huge response payload.
    compact_explored = []
    for row in sorted(
        ok_rows,
        key=lambda r: (r.get("ai_score", -1e12), r.get("return_pct", -1e12)),
        reverse=True,
    )[: min(300, len(ok_rows))]:
        compact_explored.append(
            {
                "run_id": row["run_id"],
                "strategy_id": row["strategy_id"],
                "strategy_name": row["strategy_name"],
                "timeframe": row["timeframe"],
                "return_pct": row.get("return_pct"),
                "profit_factor": row.get("profit_factor"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "trades": row.get("trades"),
                "win_rate_pct": row.get("win_rate_pct"),
                "quality_ok": row.get("quality_ok"),
                "ai_score": row.get("ai_score"),
            }
        )

    emit(1.0, "completed")
    final_result = {
        "kind": "ai_optimize",
        "top_picks": top_picks,
        "leaderboard": top_picks,
        "explored": compact_explored,
        "summary": {
            "runs_planned": total_planned_runs,
            "runs_total": len(explored),
            "runs_ok": len(ok_rows),
            "quality_ok": len(quality_rows),
            "objective": objective,
            "top_n": top_n,
            "min_trades": min_trades,
            "max_drawdown_pct": max_drawdown_abs_pct,
            "seed": seed,
            "strategies_tested": strategy_ids,
            "timeframes_tested": sorted({p[1] for p in pairs}),
            "cached": False,
            "cache_key": cache_key,
        },
    }
    _write_ai_cache(
        cache_key,
        {
            "top_picks": top_picks,
            "explored": compact_explored,
            "summary": final_result["summary"],
        },
    )
    return final_result


def describe_timeframes(db: str = DEFAULT_DB) -> dict[str, Any]:
    rows = []
    for timeframe in ALLOWED_TIMEFRAMES:
        start_ms, end_ms = resolve_dataset_range(Path(db), timeframe)
        rows.append(
            {
                "timeframe": timeframe,
                "available": start_ms is not None and end_ms is not None,
                "start": ms_to_iso(start_ms) if start_ms else None,
                "end": ms_to_iso(end_ms) if end_ms else None,
            }
        )
    return {"db": db, "items": rows}


def list_symbols(db: str = DEFAULT_DB, timeframe: str = "1h", limit: int = 30) -> list[dict[str, Any]]:
    if timeframe not in ALLOWED_TIMEFRAMES:
        raise ValueError(f"timeframe invalide: {timeframe}")
    db_path = Path(db)
    if not db_path.exists():
        raise ValueError(f"DB introuvable: {db}")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT
                c.symbol,
                COALESCE(MAX(tm.volume_24h), 0) AS volume_24h,
                COUNT(*) AS candles_count
            FROM candles c
            LEFT JOIN token_meta tm
              ON tm.source = c.source AND tm.symbol = c.symbol
            WHERE c.source = 'hyperliquid' AND c.interval = ?
            GROUP BY c.symbol
            ORDER BY volume_24h DESC, candles_count DESC
            LIMIT ?
            """,
            (timeframe, max(1, min(500, int(limit)))),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "symbol": str(row[0]),
            "volume_24h": float(row[1] or 0.0),
            "candles_count": int(row[2] or 0),
        }
        for row in rows
    ]


def export_blotter(result: dict[str, Any], limit: int, offset: int) -> dict[str, Any]:
    trades_df = result.get("trades_df")
    if not isinstance(trades_df, pd.DataFrame):
        raise ValueError("Aucun blotter pour ce job")
    ordered = trades_df.sort_values("exit_ts_ms", ascending=False).reset_index(drop=True)
    total, rows = _clean_records(ordered, limit=limit, offset=offset)
    return {"total": total, "items": rows}


def export_equity_curve(result: dict[str, Any], limit: int, offset: int) -> dict[str, Any]:
    equity_df = result.get("equity_df")
    if not isinstance(equity_df, pd.DataFrame):
        raise ValueError("Aucune courbe equity pour ce job")
    total, rows = _clean_records(equity_df, limit=limit, offset=offset)
    return {"total": total, "items": rows}
