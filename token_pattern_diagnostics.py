#!/usr/bin/env python3
"""
Diagnostic des patterns par token pour la strategie squeeze.

Objectif:
- Comprendre pourquoi certains tokens performent mieux
- Extraire des patterns de features potentiellement reproductibles

Usage:
  python3 token_pattern_diagnostics.py --db squeeze_data.db
  python3 token_pattern_diagnostics.py --max-tokens 40 --min-token-trades 25
  python3 token_pattern_diagnostics.py --export-token-summary token_summary.csv --export-rules rules.csv
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_hyperliquid import (
    compute_max_drawdown,
    compute_profit_factor,
    load_candles,
    load_symbols,
    parse_optional_date,
)
from squeeze_detector import (
    SqueezeConfig,
    backtest_squeeze_signals,
    compute_indicators,
    compute_squeeze_score,
    estimate_expected_move,
    predict_direction,
)


FEATURE_COLS = [
    "entry_score",
    "entry_expected_move",
    "entry_bb_width_pct",
    "entry_atr_pct_rank",
    "entry_ttm_bars",
    "entry_vol_ratio",
    "entry_rsi",
    "entry_ema_spread",
    "entry_ema_trend_slope",
    "entry_ret_8",
    "entry_macd_hist_slope",
]


def enrich_trades_with_features(
    symbol: str,
    display_name: str,
    df_raw: pd.DataFrame,
    config: SqueezeConfig,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades

    df_ind = compute_indicators(df_raw, config)
    enriched_rows: list[dict] = []

    for _, tr in trades.iterrows():
        entry_idx = int(tr["entry_idx"])
        exit_idx = int(tr["exit_idx"])
        if entry_idx >= len(df_ind) or exit_idx >= len(df_ind):
            continue

        erow = df_ind.iloc[entry_idx]
        xrow = df_ind.iloc[exit_idx]
        direction_str = str(tr.get("direction", "unknown"))
        direction = None
        if direction_str == "long":
            direction = "long"
        elif direction_str == "short":
            direction = "short"
        else:
            direction = "unknown"

        pred_dir, pred_conf = predict_direction(erow, config)
        score = compute_squeeze_score(erow, config)

        enriched_rows.append(
            {
                "symbol": symbol,
                "display_name": display_name,
                "entry_time": df_raw.index[entry_idx],
                "exit_time": df_raw.index[exit_idx],
                "direction": direction,
                "predicted_direction": pred_dir.value,
                "predicted_confidence": pred_conf,
                "entry_price": float(tr["entry_price"]),
                "exit_price": float(tr["exit_price"]),
                "holding_bars": int(tr["holding_bars"]),
                "exit_reason": str(tr["exit_reason"]),
                "pnl_pct": float(tr["pnl_pct"]),
                "net_pnl": float(tr["net_pnl"]),
                "is_win": float(tr["net_pnl"] > 0),
                "entry_score": score,
                "entry_expected_move": estimate_expected_move(erow),
                "entry_bb_width_pct": float(erow.get("bb_width_pct", np.nan)),
                "entry_atr_pct_rank": float(erow.get("atr_pct_rank", np.nan)),
                "entry_ttm_bars": float(erow.get("ttm_squeeze_bars", np.nan)),
                "entry_vol_ratio": float(erow.get("vol_ratio", np.nan)),
                "entry_rsi": float(erow.get("rsi", np.nan)),
                "entry_ema_spread": float(erow.get("ema_spread", np.nan)),
                "entry_ema_trend_slope": float(erow.get("ema_trend_slope", np.nan)),
                "entry_ret_8": float(erow.get("ret_8", np.nan)),
                "entry_macd_hist_slope": float(erow.get("macd_hist_slope", np.nan)),
                "entry_breakout_up": int(bool(erow.get("breakout_up", False))),
                "entry_breakout_down": int(bool(erow.get("breakout_down", False))),
                "exit_vol_ratio": float(xrow.get("vol_ratio", np.nan)),
                "exit_atr_pct_rank": float(xrow.get("atr_pct_rank", np.nan)),
            }
        )

    return pd.DataFrame(enriched_rows)


def build_token_summary(trades_enriched: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades_enriched.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for symbol, g in trades_enriched.groupby("symbol"):
        g = g.sort_values("entry_time").reset_index(drop=True)
        wins = g[g["net_pnl"] > 0]
        losses = g[g["net_pnl"] <= 0]
        trades_count = len(g)
        total_pnl = float(g["net_pnl"].sum())
        final_equity = initial_capital + total_pnl
        win_rate = float((g["net_pnl"] > 0).mean())
        avg_win = float(wins["net_pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["net_pnl"].mean()) if len(losses) else 0.0
        expectancy = float(g["net_pnl"].mean()) if trades_count else 0.0
        pf = compute_profit_factor(g.rename(columns={"net_pnl": "net_pnl"}))
        equity_curve = initial_capital + g["net_pnl"].cumsum()
        max_dd = compute_max_drawdown(equity_curve, initial_capital)
        # Score de robustesse simple: favorise expectancy stable et penalise DD
        robustness_score = expectancy * np.sqrt(trades_count) - abs(max_dd * 100) * 0.5

        rows.append(
            {
                "symbol": symbol,
                "display_name": g["display_name"].iloc[0],
                "trades": trades_count,
                "total_pnl": total_pnl,
                "return_pct": (final_equity / initial_capital - 1.0) * 100.0,
                "win_rate_pct": win_rate * 100.0,
                "profit_factor": pf,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "expectancy": expectancy,
                "max_drawdown_pct": max_dd * 100.0,
                "avg_holding_bars": float(g["holding_bars"].mean()),
                "avg_entry_score": float(g["entry_score"].mean()),
                "avg_entry_expected_move": float(g["entry_expected_move"].mean()),
                "robustness_score": float(robustness_score),
            }
        )

    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)


def feature_effects_by_token_class(
    trades_enriched: pd.DataFrame,
    token_summary: pd.DataFrame,
    min_token_trades: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if trades_enriched.empty or token_summary.empty:
        return pd.DataFrame(), [], []

    strong_tokens = token_summary[
        (token_summary["trades"] >= min_token_trades)
        & (token_summary["profit_factor"] > 1.05)
        & (token_summary["return_pct"] > 0)
    ]["symbol"]
    weak_tokens = token_summary[
        (token_summary["trades"] >= min_token_trades)
        & ((token_summary["profit_factor"] < 0.95) | (token_summary["return_pct"] < 0))
    ]["symbol"]

    strong = trades_enriched[trades_enriched["symbol"].isin(set(strong_tokens))].copy()
    weak = trades_enriched[trades_enriched["symbol"].isin(set(weak_tokens))].copy()
    if strong.empty or weak.empty:
        return pd.DataFrame(), list(strong_tokens), list(weak_tokens)

    rows: list[dict] = []
    for feat in FEATURE_COLS:
        s = strong[feat].replace([np.inf, -np.inf], np.nan).dropna()
        w = weak[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20 or len(w) < 20:
            continue
        s_mean = float(s.mean())
        w_mean = float(w.mean())
        s_std = float(s.std(ddof=0))
        w_std = float(w.std(ddof=0))
        pooled = np.sqrt((s_std**2 + w_std**2) / 2.0) if (s_std > 0 or w_std > 0) else 0.0
        effect = (s_mean - w_mean) / pooled if pooled > 0 else 0.0

        rows.append(
            {
                "feature": feat,
                "strong_mean": s_mean,
                "weak_mean": w_mean,
                "delta": s_mean - w_mean,
                "effect_size": effect,
                "strong_n": len(s),
                "weak_n": len(w),
            }
        )

    if not rows:
        return pd.DataFrame(), list(strong_tokens), list(weak_tokens)
    return (
        pd.DataFrame(rows).sort_values("effect_size", key=lambda x: x.abs(), ascending=False),
        list(strong_tokens),
        list(weak_tokens),
    )


def mine_univariate_rules(
    trades_enriched: pd.DataFrame,
    min_rule_trades: int,
    min_rule_tstat: float,
    top_rules: int,
) -> pd.DataFrame:
    if trades_enriched.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    global_expectancy = float(trades_enriched["net_pnl"].mean())
    global_win_rate = float((trades_enriched["net_pnl"] > 0).mean())

    quantiles = [0.25, 0.5, 0.75]
    for feat in FEATURE_COLS:
        series = trades_enriched[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < max(3 * min_rule_trades, 120):
            continue
        qvals = sorted(series.quantile(quantiles).unique())
        for q in qvals:
            subset_ge = trades_enriched[trades_enriched[feat] >= q]
            subset_le = trades_enriched[trades_enriched[feat] <= q]

            for op, subset in ((">=", subset_ge), ("<=", subset_le)):
                n = len(subset)
                if n < min_rule_trades:
                    continue
                expectancy = float(subset["net_pnl"].mean())
                std = float(subset["net_pnl"].std(ddof=1)) if n > 1 else 0.0
                win_rate = float((subset["net_pnl"] > 0).mean())
                lift = expectancy - global_expectancy
                wr_lift = win_rate - global_win_rate
                stderr = (std / np.sqrt(n)) if std > 0 else 0.0
                tstat = (lift / stderr) if stderr > 0 else 0.0

                if abs(tstat) < min_rule_tstat:
                    continue

                rows.append(
                    {
                        "feature": feat,
                        "operator": op,
                        "threshold": float(q),
                        "trades": n,
                        "expectancy": expectancy,
                        "win_rate_pct": win_rate * 100.0,
                        "expectancy_lift": lift,
                        "win_rate_lift_pp": wr_lift * 100.0,
                        "lift_tstat": tstat,
                    }
                )

    if not rows:
        return pd.DataFrame()

    rules = pd.DataFrame(rows)
    rules = rules.sort_values(
        ["expectancy_lift", "win_rate_lift_pp", "trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return rules.head(top_rules)


def run() -> int:
    parser = argparse.ArgumentParser(description="Diagnose reproducible token patterns for squeeze strategy")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--source", type=str, default="hyperliquid")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--min-candles", type=int, default=500)
    parser.add_argument("--min-volume", type=float, default=100_000)
    parser.add_argument("--capital-per-token", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--position-pct", type=float, default=0.5)
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--target-atr", type=float, default=3.0)
    parser.add_argument("--trailing-stop", type=float, default=0.015)
    parser.add_argument("--taker-fee", type=float, default=0.00035)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--min-squeeze-score", type=float, default=0.45)
    parser.add_argument("--ready-score", type=float, default=0.70)
    parser.add_argument("--firing-score", type=float, default=0.50)
    parser.add_argument("--min-token-trades", type=int, default=25)
    parser.add_argument("--min-rule-trades", type=int, default=120)
    parser.add_argument("--top-rules", type=int, default=12)
    parser.add_argument("--min-strong-tokens", type=int, default=3)
    parser.add_argument("--min-weak-tokens", type=int, default=3)
    parser.add_argument("--min-rule-tstat", type=float, default=1.0)
    parser.add_argument("--export-token-summary", type=str, default="")
    parser.add_argument("--export-trades-enriched", type=str, default="")
    parser.add_argument("--export-effects", type=str, default="")
    parser.add_argument("--export-rules", type=str, default="")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB introuvable: {db_path}")
        return 1

    start_ms = parse_optional_date(args.start)
    end_ms = parse_optional_date(args.end, end_of_day=True)
    if start_ms and end_ms and start_ms > end_ms:
        print("Erreur: --start doit etre <= --end")
        return 1

    config = SqueezeConfig(
        min_squeeze_score=args.min_squeeze_score,
        ready_squeeze_score=args.ready_score,
        firing_score=args.firing_score,
    )

    conn = sqlite3.connect(str(db_path))
    symbols = load_symbols(
        conn=conn,
        source=args.source,
        interval=args.interval,
        min_candles=args.min_candles,
        min_volume=args.min_volume,
        max_tokens=args.max_tokens,
    )
    if not symbols:
        print("Aucun token eligibile.")
        return 1

    per_token_rows: list[dict] = []
    all_trades_enriched: list[pd.DataFrame] = []
    tested = 0
    skipped = 0

    for symbol, display_name, volume_24h, candles_count in symbols:
        df = load_candles(
            conn=conn,
            source=args.source,
            symbol=symbol,
            interval=args.interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if df.empty or len(df) < args.min_candles:
            skipped += 1
            continue

        tested += 1
        trades = backtest_squeeze_signals(
            df=df,
            config=config,
            capital=args.capital_per_token,
            leverage=args.leverage,
            position_pct=args.position_pct,
            stop_atr_mult=args.stop_atr,
            target_atr_mult=args.target_atr,
            trailing_stop_pct=args.trailing_stop,
            taker_fee=args.taker_fee,
            slippage_bps=args.slippage_bps,
        )

        if trades.empty:
            per_token_rows.append(
                {
                    "symbol": symbol,
                    "display_name": display_name,
                    "volume_24h": volume_24h,
                    "candles_count": candles_count,
                    "trades": 0,
                    "total_pnl": 0.0,
                    "return_pct": 0.0,
                    "profit_factor": 0.0,
                    "win_rate_pct": 0.0,
                    "expectancy": 0.0,
                    "max_drawdown_pct": 0.0,
                    "robustness_score": 0.0,
                }
            )
            continue

        enr = enrich_trades_with_features(symbol, display_name, df, config, trades)
        if not enr.empty:
            all_trades_enriched.append(enr)

        final_equity = float(trades["equity"].iloc[-1])
        win_rate = float((trades["net_pnl"] > 0).mean())
        profit_factor = compute_profit_factor(trades)
        max_dd = compute_max_drawdown(trades["equity"], args.capital_per_token)
        total_pnl = float(trades["net_pnl"].sum())
        expectancy = float(trades["net_pnl"].mean())
        robustness_score = expectancy * np.sqrt(len(trades)) - abs(max_dd * 100.0) * 0.5

        per_token_rows.append(
            {
                "symbol": symbol,
                "display_name": display_name,
                "volume_24h": volume_24h,
                "candles_count": candles_count,
                "trades": int(len(trades)),
                "total_pnl": total_pnl,
                "return_pct": (final_equity / args.capital_per_token - 1.0) * 100,
                "profit_factor": profit_factor,
                "win_rate_pct": win_rate * 100,
                "expectancy": expectancy,
                "max_drawdown_pct": max_dd * 100,
                "robustness_score": robustness_score,
            }
        )

    conn.close()

    token_summary = pd.DataFrame(per_token_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
    trades_enriched = pd.concat(all_trades_enriched, ignore_index=True) if all_trades_enriched else pd.DataFrame()

    effects_df, strong_tokens, weak_tokens = feature_effects_by_token_class(
        trades_enriched=trades_enriched,
        token_summary=token_summary,
        min_token_trades=args.min_token_trades,
    )
    rules_df = mine_univariate_rules(
        trades_enriched=trades_enriched,
        min_rule_trades=args.min_rule_trades,
        min_rule_tstat=args.min_rule_tstat,
        top_rules=args.top_rules,
    )

    print("=" * 100)
    print("TOKEN PATTERN DIAGNOSTICS")
    print("=" * 100)
    print(f"DB: {args.db}")
    print(f"Fenetre: {args.start or 'debut'} -> {args.end or 'fin'}")
    print(f"Tokens selectionnes={len(symbols)} | testes={tested} | skip={skipped}")
    print(f"Trades analyses={len(trades_enriched)}")
    print("-" * 100)

    if not token_summary.empty:
        print("Top 12 tokens par robustesse (pas juste PnL):")
        cols = [
            "symbol",
            "trades",
            "total_pnl",
            "return_pct",
            "profit_factor",
            "win_rate_pct",
            "expectancy",
            "max_drawdown_pct",
            "robustness_score",
        ]
        print(token_summary.sort_values("robustness_score", ascending=False)[cols].head(12).to_string(index=False))
        robust_only = token_summary[token_summary["trades"] >= args.min_token_trades].copy()
        if not robust_only.empty:
            print("-" * 100)
            print(f"Top 12 tokens robustes (>= {args.min_token_trades} trades):")
            print(robust_only.sort_values("robustness_score", ascending=False)[cols].head(12).to_string(index=False))
        print("-" * 100)

    strong_n = len(strong_tokens)
    weak_n = len(weak_tokens)
    if strong_n >= args.min_strong_tokens and weak_n >= args.min_weak_tokens and not effects_df.empty:
        print(f"Tokens forts={strong_n} | tokens faibles={weak_n}")
        print("Features qui differencient tokens forts vs faibles (effet absolu):")
        print(
            effects_df[
                ["feature", "strong_mean", "weak_mean", "delta", "effect_size", "strong_n", "weak_n"]
            ]
            .head(12)
            .to_string(index=False)
        )
        print("-" * 100)
    else:
        print(
            "Pas assez de donnees robustes pour comparer tokens forts vs faibles "
            f"(forts={strong_n}, faibles={weak_n})."
        )
        print("-" * 100)

    if not rules_df.empty:
        print("Patterns potentiellement reproductibles (regles univariées):")
        print(
            rules_df[
                [
                    "feature",
                    "operator",
                    "threshold",
                    "trades",
                    "expectancy",
                    "expectancy_lift",
                    "lift_tstat",
                    "win_rate_pct",
                    "win_rate_lift_pp",
                ]
            ].to_string(index=False)
        )
    else:
        print("Pas assez de support statistique pour extraire des regles robustes.")

    if args.export_token_summary:
        token_summary.to_csv(Path(args.export_token_summary), index=False)
        print(f"\nExport token summary: {args.export_token_summary}")
    if args.export_trades_enriched and not trades_enriched.empty:
        trades_enriched.to_csv(Path(args.export_trades_enriched), index=False)
        print(f"Export trades enriched: {args.export_trades_enriched}")
    if args.export_effects and not effects_df.empty:
        effects_df.to_csv(Path(args.export_effects), index=False)
        print(f"Export effects: {args.export_effects}")
    if args.export_rules and not rules_df.empty:
        rules_df.to_csv(Path(args.export_rules), index=False)
        print(f"Export rules: {args.export_rules}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
