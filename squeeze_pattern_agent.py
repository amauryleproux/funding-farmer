#!/usr/bin/env python3
"""
Squeeze Pattern Analyzer Agent for HyperPulse signals.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from rich.console import Console
    from rich.markdown import Markdown
except ImportError:
    Console = None
    Markdown = None


DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


class SqueezePatternAgent:
    def __init__(self, db_path: str, no_llm: bool = False, verbose: bool = False) -> None:
        self.db_path = db_path
        self.no_llm = no_llm
        self.verbose = verbose
        self.console = Console() if Console else None
        self.scratchpad: dict[str, Any] = {
            "metadata": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "db_path": db_path,
                "agent": "SqueezePatternAgent",
            },
            "tools": [],
        }
        self._resolved_total = 0
        self._expired_ignored = 0

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rate(numerator: float, denominator: float) -> float | None:
        if denominator == 0:
            return None
        return numerator / denominator

    def _append_tool_result(self, name: str, result: dict[str, Any]) -> None:
        payload = {"tool": name, "result": result}
        self.scratchpad["tools"].append(payload)
        if self.verbose:
            print(f"\n--- {name} ---")
            print(json.dumps(result, indent=2, ensure_ascii=False))

    def _print(self, text: str) -> None:
        print(text)

    def _display_markdown(self, content: str) -> None:
        if self.console and Markdown:
            self.console.print(Markdown(content))
        else:
            print(content)

    def load_resolved_trades(self, db_path: str) -> pd.DataFrame:
        """
        Charge tous les trades resolved (win + loss, ignore expired)
        et ajoute les colonnes derivees.
        """
        db_file = Path(db_path).expanduser()
        if not db_file.exists():
            raise FileNotFoundError(f"Database not found: {db_file}")

        db_uri = f"file:{db_file.resolve()}?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            df_all = pd.read_sql_query("SELECT * FROM signals WHERE resolved = 1", conn)

        self._resolved_total = int(len(df_all))
        if "result" not in df_all.columns:
            self._expired_ignored = 0
            return pd.DataFrame(columns=df_all.columns)

        result_series = df_all["result"].astype(str).str.lower()
        self._expired_ignored = int((result_series == "expired").sum())
        df = df_all[result_series.isin(["win", "loss"])].copy()

        if df.empty:
            return df

        numeric_cols = [
            "score",
            "confidence",
            "bb_width_pct",
            "atr_value",
            "volume_ratio",
            "funding_rate",
            "expected_move",
            "entry_price",
            "target_price",
            "stop_price",
            "pnl_pct",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        timestamp_col = pd.to_numeric(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(np.nan, index=df.index)
        dt = pd.to_datetime(timestamp_col, unit="s", utc=True, errors="coerce")
        df["hour_utc"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek

        if {"entry_price", "target_price", "stop_price"}.issubset(df.columns):
            denom = df["entry_price"] - df["stop_price"]
            numer = df["target_price"] - df["entry_price"]
            df["risk_reward"] = np.where(denom != 0, numer / denom, np.nan)
        else:
            df["risk_reward"] = np.nan

        if "funding_rate" in df.columns:
            df["funding_abs"] = df["funding_rate"].abs()
        else:
            df["funding_abs"] = np.nan

        return df

    def compute_feature_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        features = [
            "score",
            "confidence",
            "bb_width_pct",
            "atr_value",
            "volume_ratio",
            "funding_rate",
            "funding_abs",
            "expected_move",
        ]
        wins_df = df[df["result"] == "win"]
        losses_df = df[df["result"] == "loss"]

        output: dict[str, Any] = {"feature_count": 0, "features": {}}
        for feature in features:
            if feature not in df.columns:
                output["features"][feature] = {"missing": True}
                continue

            win_values = pd.to_numeric(wins_df[feature], errors="coerce").dropna()
            loss_values = pd.to_numeric(losses_df[feature], errors="coerce").dropna()

            p_value = None
            if len(win_values) > 0 and len(loss_values) > 0:
                try:
                    p_value = stats.mannwhitneyu(win_values, loss_values, alternative="two-sided").pvalue
                except ValueError:
                    p_value = None

            output["features"][feature] = {
                "n_wins": int(len(win_values)),
                "n_losses": int(len(loss_values)),
                "mean_wins": self._safe_float(win_values.mean()) if len(win_values) else None,
                "mean_losses": self._safe_float(loss_values.mean()) if len(loss_values) else None,
                "median_wins": self._safe_float(win_values.median()) if len(win_values) else None,
                "median_losses": self._safe_float(loss_values.median()) if len(loss_values) else None,
                "p_value": self._safe_float(p_value),
                "strong_signal": bool(p_value is not None and p_value < 0.05),
            }
            output["feature_count"] += 1

        return output

    def find_best_thresholds(self, df: pd.DataFrame) -> dict[str, Any]:
        features = [
            "score",
            "confidence",
            "bb_width_pct",
            "atr_value",
            "volume_ratio",
            "funding_rate",
            "funding_abs",
            "expected_move",
        ]
        percentiles = list(range(10, 100, 10))
        results: dict[str, Any] = {}

        for feature in features:
            if feature not in df.columns:
                results[feature] = {"missing": True}
                continue

            sample = df[[feature, "result"]].copy()
            sample[feature] = pd.to_numeric(sample[feature], errors="coerce")
            sample = sample.dropna(subset=[feature])
            if sample.empty:
                results[feature] = {"insufficient_data": True}
                continue

            sample["is_win"] = (sample["result"] == "win").astype(int)
            thresholds = np.percentile(sample[feature], percentiles)
            thresholds = sorted({float(x) for x in thresholds})

            best: dict[str, Any] | None = None
            for threshold in thresholds:
                above = sample[sample[feature] >= threshold]
                below = sample[sample[feature] < threshold]

                n_above = int(len(above))
                n_below = int(len(below))
                win_rate_above = self._safe_float(above["is_win"].mean()) if n_above else None
                win_rate_below = self._safe_float(below["is_win"].mean()) if n_below else None

                score_above = win_rate_above if win_rate_above is not None else -1.0
                score_below = win_rate_below if win_rate_below is not None else -1.0
                threshold_score = max(score_above, score_below)
                side = "above" if score_above >= score_below else "below"
                side_n = n_above if side == "above" else n_below

                candidate = {
                    "threshold": self._safe_float(threshold),
                    "win_rate_above": win_rate_above,
                    "win_rate_below": win_rate_below,
                    "n_above": n_above,
                    "n_below": n_below,
                    "best_side": side,
                    "best_side_win_rate": self._safe_float(threshold_score),
                    "best_side_n": side_n,
                }

                if best is None:
                    best = candidate
                    continue

                current_score = best["best_side_win_rate"] if best["best_side_win_rate"] is not None else -1.0
                if (
                    threshold_score > current_score
                    or (threshold_score == current_score and side_n > best["best_side_n"])
                ):
                    best = candidate

            results[feature] = best or {"insufficient_data": True}

        return {"thresholds": results, "percentiles_tested": percentiles}

    def analyze_time_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        overall_win_rate = self._safe_float((df["result"] == "win").mean())
        cutoff = (overall_win_rate + 0.10) if overall_win_rate is not None else None

        hour_stats = []
        if "hour_utc" in df.columns:
            grouped = df.groupby("hour_utc", dropna=True)["result"]
            for hour, series in grouped:
                count = int(series.count())
                wins = int((series == "win").sum())
                wr = self._rate(wins, count)
                hour_stats.append(
                    {
                        "hour_utc": self._safe_int(hour),
                        "count": count,
                        "wins": wins,
                        "win_rate": self._safe_float(wr),
                    }
                )
        hour_stats = sorted(hour_stats, key=lambda x: (x["hour_utc"] is None, x["hour_utc"]))

        day_stats = []
        if "day_of_week" in df.columns:
            grouped = df.groupby("day_of_week", dropna=True)["result"]
            for day, series in grouped:
                count = int(series.count())
                wins = int((series == "win").sum())
                wr = self._rate(wins, count)
                day_int = self._safe_int(day)
                day_stats.append(
                    {
                        "day_of_week": day_int,
                        "day_name": DAY_NAMES.get(day_int, str(day_int)),
                        "count": count,
                        "wins": wins,
                        "win_rate": self._safe_float(wr),
                    }
                )
        day_stats = sorted(day_stats, key=lambda x: (x["day_of_week"] is None, x["day_of_week"]))

        strong_hours = []
        strong_days = []
        if cutoff is not None:
            strong_hours = [row for row in hour_stats if row["win_rate"] is not None and row["win_rate"] > cutoff]
            strong_days = [row for row in day_stats if row["win_rate"] is not None and row["win_rate"] > cutoff]

        return {
            "overall_win_rate": overall_win_rate,
            "cutoff_for_strong_slot": self._safe_float(cutoff),
            "by_hour_utc": hour_stats,
            "by_day_of_week": day_stats,
            "strong_hours": strong_hours,
            "strong_days": strong_days,
        }

    def analyze_coin_performance(self, df: pd.DataFrame) -> dict[str, Any]:
        if "coin" not in df.columns:
            return {"coins": []}

        rows = []
        grouped = df.groupby("coin", dropna=True)
        for coin, group in grouped:
            count = int(len(group))
            wins = int((group["result"] == "win").sum())
            rows.append(
                {
                    "coin": str(coin),
                    "count": count,
                    "wins": wins,
                    "losses": int((group["result"] == "loss").sum()),
                    "win_rate": self._safe_float(self._rate(wins, count)),
                    "avg_pnl_pct": self._safe_float(pd.to_numeric(group.get("pnl_pct"), errors="coerce").mean()),
                }
            )

        rows.sort(key=lambda x: ((x["win_rate"] is not None), x["win_rate"], x["count"]), reverse=True)
        return {"coins": rows}

    def analyze_direction_bias(self, df: pd.DataFrame) -> dict[str, Any]:
        if "direction" not in df.columns:
            return {"directions": []}

        rows = []
        grouped = df.groupby(df["direction"].astype(str).str.lower(), dropna=True)
        for direction, group in grouped:
            count = int(len(group))
            wins = int((group["result"] == "win").sum())
            rows.append(
                {
                    "direction": direction,
                    "count": count,
                    "wins": wins,
                    "losses": int((group["result"] == "loss").sum()),
                    "win_rate": self._safe_float(self._rate(wins, count)),
                    "avg_pnl_pct": self._safe_float(pd.to_numeric(group.get("pnl_pct"), errors="coerce").mean()),
                }
            )
        rows.sort(key=lambda x: x["direction"])
        return {"directions": rows}

    def analyze_funding_alignment(self, df: pd.DataFrame) -> dict[str, Any]:
        if "funding_aligned" not in df.columns:
            return {"groups": []}

        series = pd.to_numeric(df["funding_aligned"], errors="coerce").fillna(0).astype(int)
        working = df.copy()
        working["funding_aligned_flag"] = series

        rows = []
        grouped = working.groupby("funding_aligned_flag", dropna=True)
        for flag, group in grouped:
            count = int(len(group))
            wins = int((group["result"] == "win").sum())
            rows.append(
                {
                    "funding_aligned": int(flag),
                    "count": count,
                    "wins": wins,
                    "losses": int((group["result"] == "loss").sum()),
                    "win_rate": self._safe_float(self._rate(wins, count)),
                    "avg_pnl_pct": self._safe_float(pd.to_numeric(group.get("pnl_pct"), errors="coerce").mean()),
                }
            )
        rows.sort(key=lambda x: x["funding_aligned"], reverse=True)
        return {"groups": rows}

    def generate_summary_stats(self, df: pd.DataFrame, analyses: dict[str, Any]) -> dict[str, Any]:
        n_total = int(len(df))
        n_wins = int((df["result"] == "win").sum())
        n_losses = int((df["result"] == "loss").sum())
        win_rate = self._safe_float(self._rate(n_wins, n_total))

        feature_stats = analyses.get("compute_feature_stats", {}).get("features", {})
        strong_features = []
        for feature_name, data in feature_stats.items():
            if isinstance(data, dict) and data.get("strong_signal"):
                strong_features.append(
                    {
                        "feature": feature_name,
                        "p_value": data.get("p_value"),
                        "mean_wins": data.get("mean_wins"),
                        "mean_losses": data.get("mean_losses"),
                    }
                )

        thresholds = analyses.get("find_best_thresholds", {}).get("thresholds", {})
        threshold_candidates = []
        for feature_name, data in thresholds.items():
            if not isinstance(data, dict):
                continue
            if data.get("insufficient_data") or data.get("missing"):
                continue
            side = data.get("best_side")
            side_wr = data.get("best_side_win_rate")
            side_n = data.get("best_side_n")
            if side_wr is None:
                continue
            threshold_candidates.append(
                {
                    "feature": feature_name,
                    "threshold": data.get("threshold"),
                    "best_side": side,
                    "best_side_win_rate": side_wr,
                    "best_side_n": side_n,
                    "delta_vs_overall": self._safe_float(side_wr - win_rate) if win_rate is not None else None,
                }
            )
        threshold_candidates.sort(
            key=lambda x: (
                x["delta_vs_overall"] is not None,
                x["delta_vs_overall"] if x["delta_vs_overall"] is not None else -999,
                x["best_side_n"] if x["best_side_n"] is not None else 0,
            ),
            reverse=True,
        )

        top_coins = analyses.get("analyze_coin_performance", {}).get("coins", [])[:10]
        direction_bias = analyses.get("analyze_direction_bias", {}).get("directions", [])
        funding_alignment = analyses.get("analyze_funding_alignment", {}).get("groups", [])
        time_patterns = analyses.get("analyze_time_patterns", {})

        return {
            "dataset": {
                "n_total": n_total,
                "n_wins": n_wins,
                "n_losses": n_losses,
                "win_rate": win_rate,
                "resolved_total_in_db": self._resolved_total,
                "expired_ignored": self._expired_ignored,
            },
            "strong_features": strong_features,
            "top_threshold_candidates": threshold_candidates[:8],
            "top_coins": top_coins,
            "direction_bias": direction_bias,
            "funding_alignment": funding_alignment,
            "strong_hours": time_patterns.get("strong_hours", []),
            "strong_days": time_patterns.get("strong_days", []),
        }

    def _build_local_markdown_report(self, analyses: dict[str, Any], summary: dict[str, Any], llm_note: str) -> str:
        dataset = summary["dataset"]
        wr_text = f"{dataset['win_rate']:.1%}" if dataset["win_rate"] is not None else "n/a"
        lines = [
            "# SQUEEZE PATTERN ANALYSIS REPORT",
            "",
            f"_Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
            "",
            f"Dataset: {dataset['n_total']} trades resolved ({dataset['n_wins']} wins, {dataset['n_losses']} losses)",
            f"Win rate actuel: {wr_text}",
            "",
            "## Findings cles",
        ]

        strong_features = summary.get("strong_features", [])
        if strong_features:
            for item in strong_features:
                lines.append(
                    "- `{}` est discriminant (p={:.4f}, mean wins={:.4f}, mean losses={:.4f})".format(
                        item["feature"],
                        item["p_value"] if item["p_value"] is not None else float("nan"),
                        item["mean_wins"] if item["mean_wins"] is not None else float("nan"),
                        item["mean_losses"] if item["mean_losses"] is not None else float("nan"),
                    )
                )
        else:
            lines.append("- Aucun signal statistique fort (p < 0.05) detecte sur les features testees.")

        top_coins = summary.get("top_coins", [])[:5]
        if top_coins:
            lines.append("")
            lines.append("Coins les plus performants (win rate):")
            for coin in top_coins:
                wr = coin["win_rate"]
                wr_display = f"{wr:.1%}" if wr is not None else "n/a"
                lines.append(f"- {coin['coin']}: {wr_display} ({coin['count']} trades)")

        lines.append("")
        lines.append("## Recommandations de filtres")
        threshold_candidates = summary.get("top_threshold_candidates", [])
        if threshold_candidates:
            for cand in threshold_candidates[:5]:
                delta = cand["delta_vs_overall"]
                delta_display = f"{delta:+.1%}" if delta is not None else "n/a"
                side = ">=" if cand["best_side"] == "above" else "<"
                lines.append(
                    f"- Tester filtre `{cand['feature']} {side} {cand['threshold']:.6g}` "
                    f"(win rate segment={cand['best_side_win_rate']:.1%}, delta={delta_display}, n={cand['best_side_n']})"
                )
        else:
            lines.append("- Pas de seuil robuste trouve avec les percentiles 10..90.")

        direction_rows = summary.get("direction_bias", [])
        if direction_rows:
            lines.append("- Verifier un biais directionnel:")
            for row in direction_rows:
                wr = row["win_rate"]
                wr_display = f"{wr:.1%}" if wr is not None else "n/a"
                lines.append(f"- Direction {row['direction']}: {wr_display} ({row['count']} trades)")

        funding_rows = summary.get("funding_alignment", [])
        if funding_rows:
            lines.append("- Verifier le filtre funding_aligned:")
            for row in funding_rows:
                wr = row["win_rate"]
                wr_display = f"{wr:.1%}" if wr is not None else "n/a"
                lines.append(f"- funding_aligned={row['funding_aligned']}: {wr_display} ({row['count']} trades)")

        lines.append("")
        lines.append("## Alertes de biais")
        lines.append("- Risque d'overfit eleve si les filtres sont bases sur peu d'echantillons.")
        if dataset["n_total"] < 20:
            lines.append("- Dataset tres petit (<20), conclusions indicatives seulement.")
        if summary.get("strong_hours") or summary.get("strong_days"):
            lines.append("- Les patterns horaires/journaliers peuvent refleter un biais de calendrier temporaire.")
        else:
            lines.append("- Aucun creneau horaire/journalier clairement au-dessus de la moyenne +10%.")

        lines.append("")
        lines.append(f"_Note: {llm_note}_")
        lines.append("")
        lines.append("## Annexes (JSON)")
        lines.append("```json")
        lines.append(json.dumps(analyses, indent=2, ensure_ascii=False))
        lines.append("```")
        return "\n".join(lines)

    def _call_llm_synthesis(self, scratchpad_json: str, n_total: int, n_wins: int, n_losses: int, win_rate: float | None) -> str | None:
        if self.no_llm:
            self._print("[INFO] LLM disabled via --no-llm. Running stats-only mode.")
            return None

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self._print("[WARN] ANTHROPIC_API_KEY not set. Skipping LLM synthesis.")
            self._print("   Set it with: export ANTHROPIC_API_KEY='your_api_key'")
            return None
        if anthropic is None:
            self._print("[WARN] Package 'anthropic' not installed. Skipping LLM synthesis.")
            self._print("   Install with: pip install anthropic")
            return None

        system_prompt = (
            "Tu es un analyste quantitatif specialise en trading crypto algorithmique.\n"
            "Tu analyses les resultats statistiques d'un bot de detection de squeezes sur Hyperliquid.\n"
            "Tu dois formuler des recommandations CONCRETES et ACTIONNABLES pour ameliorer le filtre de signaux.\n"
            "Sois precis sur les seuils (ex: \"augmenter min_score de 0.55 a 0.72\").\n"
            "Formatte ta reponse en markdown avec sections : Findings cles, Recommandations de filtres, Alertes de biais."
        )
        wr_str = f"{win_rate:.1%}" if win_rate is not None else "n/a"
        user_prompt = (
            "Voici les resultats statistiques complets de l'analyse :\n"
            f"{scratchpad_json}\n\n"
            f"Dataset : {n_total} trades resolved ({n_wins} wins, {n_losses} losses)\n"
            f"Win rate actuel : {wr_str}\n"
        )

        self._print("\n[LLM] Calling Claude for synthesis...")
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            self._print(f"[WARN] LLM call failed: {exc}")
            return None

        chunks = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    chunks.append(text)

        synthesized = "\n".join(chunks).strip()
        return synthesized or None

    def run(self) -> int:
        self._print(f"[LOAD] Loading resolved trades from {self.db_path}...")
        try:
            df = self.load_resolved_trades(self.db_path)
        except Exception as exc:
            self._print(f"[ERROR] Failed to load data: {exc}")
            return 1

        n_total = int(len(df))
        n_wins = int((df["result"] == "win").sum()) if n_total else 0
        n_losses = int((df["result"] == "loss").sum()) if n_total else 0
        win_rate = self._rate(n_wins, n_total) if n_total else None
        win_rate_text = f"{win_rate:.1%}" if win_rate is not None else "n/a"

        self._print(
            f"   -> {n_total} win/loss trades retained ({n_wins} wins, {n_losses} losses), "
            f"{self._expired_ignored} expired ignored"
        )

        if n_total == 0:
            self._print("[WARN] No resolved win/loss trades found. Nothing to analyze.")
            return 1

        if n_total < 20:
            self._print("[WARN] Dataset has fewer than 20 resolved trades. Conclusions may be noisy.")

        self._print("\n[ANALYSIS] Running analysis tools...")
        analyses: dict[str, Any] = {}

        feature_stats = self.compute_feature_stats(df)
        analyses["compute_feature_stats"] = feature_stats
        self._append_tool_result("compute_feature_stats", feature_stats)
        self._print(f"   [OK] Feature stats computed ({feature_stats.get('feature_count', 0)} features)")

        thresholds = self.find_best_thresholds(df)
        analyses["find_best_thresholds"] = thresholds
        self._append_tool_result("find_best_thresholds", thresholds)
        self._print("   [OK] Optimal thresholds found")

        time_patterns = self.analyze_time_patterns(df)
        analyses["analyze_time_patterns"] = time_patterns
        self._append_tool_result("analyze_time_patterns", time_patterns)
        self._print("   [OK] Time patterns analyzed")

        coin_performance = self.analyze_coin_performance(df)
        analyses["analyze_coin_performance"] = coin_performance
        self._append_tool_result("analyze_coin_performance", coin_performance)
        self._print("   [OK] Coin performance analyzed")

        direction_bias = self.analyze_direction_bias(df)
        analyses["analyze_direction_bias"] = direction_bias
        self._append_tool_result("analyze_direction_bias", direction_bias)
        self._print("   [OK] Direction bias analyzed")

        funding_alignment = self.analyze_funding_alignment(df)
        analyses["analyze_funding_alignment"] = funding_alignment
        self._append_tool_result("analyze_funding_alignment", funding_alignment)
        self._print("   [OK] Funding alignment analyzed")

        summary_stats = self.generate_summary_stats(df, analyses)
        analyses["generate_summary_stats"] = summary_stats
        self._append_tool_result("generate_summary_stats", summary_stats)

        self.scratchpad["dataset"] = {
            "n_total": n_total,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "win_rate": win_rate,
            "resolved_total": self._resolved_total,
            "expired_ignored": self._expired_ignored,
        }

        scratchpad_json = json.dumps(self.scratchpad, indent=2, ensure_ascii=False)
        llm_markdown = self._call_llm_synthesis(
            scratchpad_json=scratchpad_json,
            n_total=n_total,
            n_wins=n_wins,
            n_losses=n_losses,
            win_rate=win_rate,
        )

        llm_note = "LLM synthesis generated with Claude." if llm_markdown else "LLM synthesis skipped."
        report_markdown = llm_markdown or self._build_local_markdown_report(analyses, summary_stats, llm_note)

        self._print("\n" + "=" * 39)
        self._print(" SQUEEZE PATTERN ANALYSIS REPORT")
        self._print("=" * 39 + "\n")
        self._display_markdown(report_markdown)
        self._print("\n" + "=" * 39)

        report_dir = Path(__file__).resolve().parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report_path = report_dir / f"squeeze_analysis_{date_str}.md"
        raw_path = report_dir / f"squeeze_raw_{date_str}.json"

        report_path.write_text(report_markdown, encoding="utf-8")
        raw_path.write_text(scratchpad_json, encoding="utf-8")

        self._print(f"[FILE] Report saved: {report_path}")
        self._print(f"[FILE] Raw data saved: {raw_path}")
        self._print(f"[INFO] Current win rate: {win_rate_text}")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Squeeze Pattern Analyzer Agent")
    parser.add_argument(
        "--db",
        default="/home/ubuntu/funding-farmer/hyperpulse.db",
        help="Path to SQLite database (default: /home/ubuntu/funding-farmer/hyperpulse.db)",
    )
    parser.add_argument("--no-llm", action="store_true", help="Run stats only, skip LLM synthesis")
    parser.add_argument("--verbose", action="store_true", help="Print each tool result JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent = SqueezePatternAgent(db_path=args.db, no_llm=args.no_llm, verbose=args.verbose)
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
