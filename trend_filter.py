"""
=============================================================================
TREND FILTER — BTC + token trend analysis to block counter-trend signals
=============================================================================
Computes trend bias from EMA alignment and price action, then decides
whether a signal should be blocked because it fights the macro trend.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrendInfo:
    """Trend analysis result for a single asset."""
    bias: str           # "bullish", "bearish", "neutral"
    strength: float     # 0-1, how strong the trend is
    ema_aligned: bool   # EMA50 > EMA200 (bull) or EMA50 < EMA200 (bear)
    price_vs_ema200: float  # % distance from EMA200


def compute_trend(df: pd.DataFrame) -> TrendInfo:
    """
    Compute trend bias from OHLCV DataFrame.

    Uses:
    - EMA 50 vs EMA 200 alignment
    - Price position relative to EMAs
    - Recent price action (last 20 candles slope)
    """
    if df is None or len(df) < 200:
        return TrendInfo(bias="neutral", strength=0.0, ema_aligned=False, price_vs_ema200=0.0)

    close = df["close"].astype(float)

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    current_price = close.iloc[-1]
    current_ema50 = ema50.iloc[-1]
    current_ema200 = ema200.iloc[-1]

    if current_ema200 == 0:
        return TrendInfo(bias="neutral", strength=0.0, ema_aligned=False, price_vs_ema200=0.0)

    price_vs_ema200 = (current_price - current_ema200) / current_ema200

    # EMA alignment
    ema50_above_200 = current_ema50 > current_ema200

    # Scoring
    bull_score = 0.0
    bear_score = 0.0

    # 1. EMA alignment (weight 3.0)
    if ema50_above_200:
        bull_score += 3.0
    else:
        bear_score += 3.0

    # 2. Price vs EMA200 (weight 2.0)
    if current_price > current_ema200:
        bull_score += 2.0
    else:
        bear_score += 2.0

    # 3. Price vs EMA50 (weight 1.5)
    if current_price > current_ema50:
        bull_score += 1.5
    else:
        bear_score += 1.5

    # 4. Recent slope of EMA50 (weight 1.5)
    if len(ema50) >= 20:
        ema50_slope = (ema50.iloc[-1] - ema50.iloc[-20]) / ema50.iloc[-20]
        if ema50_slope > 0.005:
            bull_score += 1.5
        elif ema50_slope < -0.005:
            bear_score += 1.5

    # 5. Recent candle action — majority green/red in last 10 (weight 1.0)
    if len(close) >= 10:
        last_10 = close.iloc[-10:]
        green_count = (last_10.diff().dropna() > 0).sum()
        if green_count >= 7:
            bull_score += 1.0
        elif green_count <= 3:
            bear_score += 1.0

    total = bull_score + bear_score
    if total == 0:
        return TrendInfo(bias="neutral", strength=0.0, ema_aligned=False, price_vs_ema200=price_vs_ema200)

    strength = abs(bull_score - bear_score) / total

    if strength < 0.15:
        bias = "neutral"
    elif bull_score > bear_score:
        bias = "bullish"
    else:
        bias = "bearish"

    return TrendInfo(
        bias=bias,
        strength=strength,
        ema_aligned=ema50_above_200,
        price_vs_ema200=price_vs_ema200,
    )


def should_block_signal(
    direction: str,
    token_trend: TrendInfo,
    btc_trend: TrendInfo,
) -> tuple[bool, str]:
    """
    Decide whether a signal should be blocked based on trend analysis.

    Rules:
    - LONG in bearish BTC trend with strong strength → block
    - SHORT in bullish BTC trend with strong strength → block
    - LONG in bearish token trend with strong strength → block
    - SHORT in bullish token trend with strong strength → block

    Returns: (should_block, reason)
    """
    # BTC trend filter (strongest filter)
    if btc_trend.strength >= 0.3:
        if direction == "long" and btc_trend.bias == "bearish":
            return True, f"BTC trend bearish (strength={btc_trend.strength:.2f})"
        if direction == "short" and btc_trend.bias == "bullish":
            return True, f"BTC trend bullish (strength={btc_trend.strength:.2f})"

    # Token trend filter
    if token_trend.strength >= 0.4:
        if direction == "long" and token_trend.bias == "bearish":
            return True, f"Token trend bearish (strength={token_trend.strength:.2f})"
        if direction == "short" and token_trend.bias == "bullish":
            return True, f"Token trend bullish (strength={token_trend.strength:.2f})"

    return False, ""
