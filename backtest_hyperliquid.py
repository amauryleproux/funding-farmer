#!/usr/bin/env python3
"""
Backtest de la strategie squeeze sur donnees Hyperliquid (SQLite local).

Usage:
  python3 backtest_hyperliquid.py
  python3 backtest_hyperliquid.py --db squeeze_data.db --max-tokens 30
  python3 backtest_hyperliquid.py --start 2025-01-01 --end 2025-12-31
  python3 backtest_hyperliquid.py --export-trades trades_backtest.csv
"""

import argparse
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from squeeze_detector import SqueezeConfig, backtest_squeeze_signals


def parse_optional_date(value: Optional[str], end_of_day: bool = False) -> Optional[int]:
    if not value:
        return None
    base = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        base = base.replace(hour=23, minute=59, second=59)
    return int(base.timestamp() * 1000)


def load_symbols(
    conn: sqlite3.Connection,
    source: str,
    interval: str,
    min_candles: int,
    min_volume: float,
    max_tokens: int,
) -> list[tuple[str, str, float, int]]:
    rows = conn.execute(
        """
        SELECT
            c.symbol,
            COALESCE(tm.display_name, MAX(c.display_name), c.symbol) AS display_name,
            COALESCE(MAX(tm.volume_24h), 0) AS volume_24h,
            COUNT(*) AS candles_count
        FROM candles c
        LEFT JOIN token_meta tm
          ON tm.source = c.source AND tm.symbol = c.symbol
        WHERE c.source = ? AND c.interval = ?
        GROUP BY c.symbol
        HAVING COUNT(*) >= ? AND COALESCE(MAX(tm.volume_24h), 0) >= ?
        ORDER BY volume_24h DESC, candles_count DESC
        LIMIT ?
        """,
        (source, interval, min_candles, min_volume, max_tokens),
    ).fetchall()
    return [(str(r[0]), str(r[1]), float(r[2] or 0.0), int(r[3])) for r in rows]


def load_candles(
    conn: sqlite3.Connection,
    source: str,
    symbol: str,
    interval: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> pd.DataFrame:
    query = """
        SELECT timestamp as t, open, high, low, close, volume
        FROM candles
        WHERE source = ? AND symbol = ? AND interval = ?
    """
    params: list[object] = [source, symbol, interval]

    if start_ms is not None:
        query += " AND timestamp >= ?"
        params.append(start_ms)
    if end_ms is not None:
        query += " AND timestamp <= ?"
        params.append(end_ms)

    query += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df


def compute_profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_win = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return abs(float(gross_win / gross_loss))


def compute_max_drawdown(equity: pd.Series, initial_capital: float) -> float:
    if equity.empty:
        return 0.0
    curve = pd.concat([pd.Series([initial_capital]), equity.reset_index(drop=True)], ignore_index=True)
    running_max = curve.cummax()
    dd = (curve - running_max) / running_max
    return float(dd.min())


def run() -> int:
    parser = argparse.ArgumentParser(description="Backtest squeeze strategy on Hyperliquid data")
    parser.add_argument("--db", type=str, default="squeeze_data.db")
    parser.add_argument("--source", type=str, default="hyperliquid")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--min-candles", type=int, default=500)
    parser.add_argument("--min-volume", type=float, default=100_000)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
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
    parser.add_argument("--export-trades", type=str, default="")
    parser.add_argument("--export-summary", type=str, default="")
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
        print("Aucun token eligibile pour le backtest avec ces filtres.")
        return 1

    per_token_rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []
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

        if not trades.empty:
            trades = trades.copy()
            trades["symbol"] = symbol
            trades["display_name"] = display_name
            trades["entry_time"] = trades["entry_idx"].apply(lambda i: df.index[int(i)])
            trades["exit_time"] = trades["exit_idx"].apply(lambda i: df.index[int(i)])
            all_trades.append(trades)

            final_equity = float(trades["equity"].iloc[-1])
            win_rate = float((trades["net_pnl"] > 0).mean())
            profit_factor = compute_profit_factor(trades)
            max_dd = compute_max_drawdown(trades["equity"], args.capital_per_token)
            total_pnl = float(trades["net_pnl"].sum())
            trades_count = int(len(trades))
        else:
            final_equity = args.capital_per_token
            win_rate = 0.0
            profit_factor = 0.0
            max_dd = 0.0
            total_pnl = 0.0
            trades_count = 0

        per_token_rows.append(
            {
                "symbol": symbol,
                "display_name": display_name,
                "volume_24h": volume_24h,
                "candles_count": candles_count,
                "bars_used": len(df),
                "trades": trades_count,
                "total_pnl": total_pnl,
                "final_equity": final_equity,
                "return_pct": (final_equity / args.capital_per_token - 1.0) * 100,
                "win_rate_pct": win_rate * 100,
                "profit_factor": profit_factor,
                "max_drawdown_pct": max_dd * 100,
                "start": str(df.index[0]),
                "end": str(df.index[-1]),
            }
        )

    summary_df = pd.DataFrame(per_token_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    total_capital = args.capital_per_token * tested
    total_final = float(summary_df["final_equity"].sum()) if not summary_df.empty else 0.0
    total_pnl = total_final - total_capital
    total_return_pct = (total_pnl / total_capital * 100) if total_capital > 0 else 0.0
    total_trades = int(summary_df["trades"].sum()) if not summary_df.empty else 0
    total_win_rate = float((trades_df["net_pnl"] > 0).mean() * 100) if not trades_df.empty else 0.0
    total_profit_factor = compute_profit_factor(trades_df) if not trades_df.empty else 0.0

    print("=" * 90)
    print("BACKTEST SQUEEZE - HYPERLIQUID")
    print("=" * 90)
    print(f"DB: {args.db}")
    print(f"Source/interval: {args.source}/{args.interval}")
    print(f"Fenetre: {args.start or 'debut'} -> {args.end or 'fin'}")
    print(
        f"Tokens selectionnes: {len(symbols)} | testes: {tested} | "
        f"skip (pas assez de bars): {skipped}"
    )
    print("-" * 90)
    print(f"Capital total deploye: ${total_capital:,.2f}")
    print(f"Capital final:         ${total_final:,.2f}")
    print(f"PnL total:             ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"Nombre de trades:      {total_trades}")
    print(f"Win rate global:       {total_win_rate:.1f}%")
    print(f"Profit factor global:  {total_profit_factor:.2f}")
    print("-" * 90)

    if not summary_df.empty:
        print("Top 10 tokens par PnL:")
        display_cols = [
            "symbol",
            "trades",
            "total_pnl",
            "return_pct",
            "win_rate_pct",
            "profit_factor",
            "max_drawdown_pct",
        ]
        print(summary_df[display_cols].head(10).to_string(index=False))

    if args.export_summary:
        summary_path = Path(args.export_summary)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary exporte: {summary_path}")

    if args.export_trades and not trades_df.empty:
        trades_path = Path(args.export_trades)
        trades_df.to_csv(trades_path, index=False)
        print(f"Trades exportes: {trades_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
