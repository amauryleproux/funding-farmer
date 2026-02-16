"""
=============================================================================
STRUCTURE ANALYSIS — Market structure analysis for signal quality
=============================================================================
Adds Fair Value Gaps, Break of Structure, Change of Character,
Order Blocks, and Liquidity Levels to improve signal quality.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FairValueGap:
    """A Fair Value Gap (imbalance) in price."""
    index: int          # candle index where FVG was created
    direction: str      # "bullish" or "bearish"
    top: float          # upper bound of the gap
    bottom: float       # lower bound of the gap
    filled: bool        # whether price has returned to fill the gap


@dataclass
class OrderBlock:
    """An Order Block (institutional supply/demand zone)."""
    index: int          # candle index of the order block
    direction: str      # "bullish" or "bearish"
    zone_top: float     # upper bound of the zone
    zone_bottom: float  # lower bound of the zone
    fresh: bool         # whether the zone has been tested


@dataclass
class LiquidityLevel:
    """A liquidity level (equal highs/lows)."""
    price: float        # the price level
    level_type: str     # "buy_side" (equal lows) or "sell_side" (equal highs)
    touches: int        # number of times price touched this level
    swept: bool         # whether price swept past this level


@dataclass
class StructureAnalysis:
    """Complete market structure analysis."""
    fvgs: list[FairValueGap] = field(default_factory=list)
    structure_bias: str = "neutral"   # "bullish", "bearish", "neutral"
    order_blocks: list[OrderBlock] = field(default_factory=list)
    liquidity_levels: list[LiquidityLevel] = field(default_factory=list)
    recent_sweep: bool = False        # liquidity sweep in last 5 candles
    structure_score: float = 0.0      # overall structure quality 0-1


# =============================================================================
# FAIR VALUE GAPS
# =============================================================================

def _find_fvgs(df: pd.DataFrame, lookback: int = 50) -> list[FairValueGap]:
    """
    Find Fair Value Gaps in the last `lookback` candles.

    Bullish FVG: candle[i-2].high < candle[i].low (gap up)
    Bearish FVG: candle[i-2].low > candle[i].high (gap down)
    """
    fvgs = []
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    start = max(2, len(df) - lookback)

    for i in range(start, len(df)):
        # Bullish FVG: gap up — candle[i-2] high < candle[i] low
        if high[i - 2] < low[i]:
            fvg = FairValueGap(
                index=i,
                direction="bullish",
                top=low[i],
                bottom=high[i - 2],
                filled=False,
            )
            # Check if any subsequent candle filled it
            for j in range(i + 1, len(df)):
                if low[j] <= fvg.top:
                    fvg.filled = True
                    break
            fvgs.append(fvg)

        # Bearish FVG: gap down — candle[i-2] low > candle[i] high
        if low[i - 2] > high[i]:
            fvg = FairValueGap(
                index=i,
                direction="bearish",
                top=low[i - 2],
                bottom=high[i],
                filled=False,
            )
            for j in range(i + 1, len(df)):
                if high[j] >= fvg.bottom:
                    fvg.filled = True
                    break
            fvgs.append(fvg)

    return fvgs


# =============================================================================
# SWING POINTS & MARKET STRUCTURE (BOS / CHoCH)
# =============================================================================

def _find_swing_points(
    df: pd.DataFrame, pivot_len: int = 5
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Find swing highs and swing lows using N-bar pivots.

    Returns: (swing_highs, swing_lows) as lists of (index, price).
    """
    high = df["high"].values
    low = df["low"].values
    swing_highs = []
    swing_lows = []

    for i in range(pivot_len, len(df) - pivot_len):
        # Swing high: highest in window
        if high[i] == max(high[i - pivot_len : i + pivot_len + 1]):
            swing_highs.append((i, float(high[i])))
        # Swing low: lowest in window
        if low[i] == min(low[i - pivot_len : i + pivot_len + 1]):
            swing_lows.append((i, float(low[i])))

    return swing_highs, swing_lows


def _determine_structure_bias(
    df: pd.DataFrame,
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
) -> str:
    """
    Determine market structure bias from BOS and CHoCH.

    BOS: price breaks above last swing high (bullish) or below last swing low (bearish)
    CHoCH: first break in opposite direction of current trend
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "neutral"

    close = df["close"].values
    current_price = close[-1]

    # Check recent structure breaks (last 20 candles)
    recent_bullish_bos = 0
    recent_bearish_bos = 0

    for idx, price in swing_highs[-5:]:
        if current_price > price:
            recent_bullish_bos += 1

    for idx, price in swing_lows[-5:]:
        if current_price < price:
            recent_bearish_bos += 1

    # Higher highs and higher lows = bullish
    last_highs = [p for _, p in swing_highs[-3:]]
    last_lows = [p for _, p in swing_lows[-3:]]

    higher_highs = all(last_highs[i] > last_highs[i - 1] for i in range(1, len(last_highs)))
    higher_lows = all(last_lows[i] > last_lows[i - 1] for i in range(1, len(last_lows)))
    lower_highs = all(last_highs[i] < last_highs[i - 1] for i in range(1, len(last_highs)))
    lower_lows = all(last_lows[i] < last_lows[i - 1] for i in range(1, len(last_lows)))

    bull_points = 0
    bear_points = 0

    if higher_highs:
        bull_points += 2
    if higher_lows:
        bull_points += 2
    if lower_highs:
        bear_points += 2
    if lower_lows:
        bear_points += 2

    bull_points += recent_bullish_bos
    bear_points += recent_bearish_bos

    if bull_points > bear_points + 1:
        return "bullish"
    elif bear_points > bull_points + 1:
        return "bearish"
    return "neutral"


# =============================================================================
# ORDER BLOCKS
# =============================================================================

def _find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> list[OrderBlock]:
    """
    Find Order Blocks in the last `lookback` candles.

    Bullish OB: last bearish candle before an impulsive move up (> 1.5 * ATR)
    Bearish OB: last bullish candle before an impulsive move down (> 1.5 * ATR)
    """
    order_blocks = []
    open_vals = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # Compute ATR for impulsive move threshold
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    start = max(1, len(df) - lookback)

    for i in range(start, len(df) - 1):
        atr_idx = min(i - 1, len(atr) - 1)
        if atr_idx < 0:
            continue
        current_atr = atr[atr_idx]
        if current_atr <= 0:
            continue

        candle_body = close[i] - open_vals[i]
        next_move = close[i + 1] - close[i] if i + 1 < len(df) else 0

        # Bullish OB: bearish candle (close < open) followed by impulsive up move
        if candle_body < 0 and next_move > 1.5 * current_atr:
            ob = OrderBlock(
                index=i,
                direction="bullish",
                zone_top=open_vals[i],
                zone_bottom=close[i],
                fresh=True,
            )
            # Check if mitigated (price returned to zone)
            for j in range(i + 2, len(df)):
                if low[j] <= ob.zone_top:
                    ob.fresh = False
                    break
            order_blocks.append(ob)

        # Bearish OB: bullish candle (close > open) followed by impulsive down move
        if candle_body > 0 and next_move < -1.5 * current_atr:
            ob = OrderBlock(
                index=i,
                direction="bearish",
                zone_top=close[i],
                zone_bottom=open_vals[i],
                fresh=True,
            )
            for j in range(i + 2, len(df)):
                if high[j] >= ob.zone_bottom:
                    ob.fresh = False
                    break
            order_blocks.append(ob)

    return order_blocks


# =============================================================================
# LIQUIDITY LEVELS
# =============================================================================

def _find_liquidity_levels(
    df: pd.DataFrame, tolerance_pct: float = 0.001, min_touches: int = 3
) -> list[LiquidityLevel]:
    """
    Find liquidity levels (equal highs/lows).

    Equal highs (within tolerance) with 3+ touches = sell-side liquidity
    Equal lows (within tolerance) with 3+ touches = buy-side liquidity
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    levels = []

    # Group equal highs
    used_highs = set()
    for i in range(len(high)):
        if i in used_highs:
            continue
        cluster = [i]
        for j in range(i + 1, len(high)):
            if j in used_highs:
                continue
            if abs(high[j] - high[i]) / max(high[i], 1e-10) <= tolerance_pct:
                cluster.append(j)
                used_highs.add(j)

        if len(cluster) >= min_touches:
            avg_price = float(np.mean([high[k] for k in cluster]))
            # Check if swept
            swept = False
            last_touch = max(cluster)
            for k in range(last_touch + 1, len(df)):
                if high[k] > avg_price * (1 + tolerance_pct):
                    swept = True
                    break
            levels.append(LiquidityLevel(
                price=avg_price,
                level_type="sell_side",
                touches=len(cluster),
                swept=swept,
            ))

    # Group equal lows
    used_lows = set()
    for i in range(len(low)):
        if i in used_lows:
            continue
        cluster = [i]
        for j in range(i + 1, len(low)):
            if j in used_lows:
                continue
            if abs(low[j] - low[i]) / max(low[i], 1e-10) <= tolerance_pct:
                cluster.append(j)
                used_lows.add(j)

        if len(cluster) >= min_touches:
            avg_price = float(np.mean([low[k] for k in cluster]))
            swept = False
            last_touch = max(cluster)
            for k in range(last_touch + 1, len(df)):
                if low[k] < avg_price * (1 - tolerance_pct):
                    swept = True
                    break
            levels.append(LiquidityLevel(
                price=avg_price,
                level_type="buy_side",
                touches=len(cluster),
                swept=swept,
            ))

    return levels


def _check_recent_sweep(
    df: pd.DataFrame, liquidity_levels: list[LiquidityLevel], lookback: int = 5
) -> bool:
    """Check if a liquidity sweep occurred in the last `lookback` candles."""
    if not liquidity_levels:
        return False

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    for level in liquidity_levels:
        if level.swept:
            continue  # already swept earlier

        for i in range(max(0, n - lookback), n):
            if level.level_type == "sell_side":
                # Price went above then came back below
                if high[i] > level.price and close[i] < level.price:
                    return True
            elif level.level_type == "buy_side":
                # Price went below then came back above
                if low[i] < level.price and close[i] > level.price:
                    return True

    return False


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_structure(df: pd.DataFrame) -> StructureAnalysis:
    """
    Analyze market structure from OHLCV DataFrame.

    Returns StructureAnalysis with:
    - fvgs: list of FairValueGap (direction, top, bottom, filled)
    - structure_bias: "bullish" | "bearish" | "neutral" (based on BOS/CHoCH)
    - order_blocks: list of OrderBlock (direction, zone_top, zone_bottom, fresh)
    - liquidity_levels: list of LiquidityLevel (price, type, touches)
    - recent_sweep: bool (liquidity sweep in last 5 candles)
    - structure_score: float 0-1 (overall structure quality)
    """
    if df is None or len(df) < 50:
        return StructureAnalysis()

    # Fair Value Gaps
    fvgs = _find_fvgs(df, lookback=50)

    # Swing points and structure bias
    swing_highs, swing_lows = _find_swing_points(df, pivot_len=5)
    structure_bias = _determine_structure_bias(df, swing_highs, swing_lows)

    # Order Blocks
    order_blocks = _find_order_blocks(df, lookback=50)

    # Liquidity Levels
    liquidity_levels = _find_liquidity_levels(df)

    # Recent sweep
    recent_sweep = _check_recent_sweep(df, liquidity_levels, lookback=5)

    # Structure score: how much confluence exists
    score = 0.0
    score_components = 0

    # Bias clarity
    if structure_bias != "neutral":
        score += 0.25
    score_components += 0.25

    # Open FVGs nearby
    open_fvgs = [f for f in fvgs if not f.filled]
    if open_fvgs:
        score += 0.20
    score_components += 0.20

    # Fresh order blocks
    fresh_obs = [ob for ob in order_blocks if ob.fresh]
    if fresh_obs:
        score += 0.25
    score_components += 0.25

    # Liquidity levels
    if liquidity_levels:
        score += 0.10
    score_components += 0.10

    # Recent sweep bonus
    if recent_sweep:
        score += 0.20
    score_components += 0.20

    structure_score = score / score_components if score_components > 0 else 0.0

    return StructureAnalysis(
        fvgs=fvgs,
        structure_bias=structure_bias,
        order_blocks=order_blocks,
        liquidity_levels=liquidity_levels,
        recent_sweep=recent_sweep,
        structure_score=structure_score,
    )


def should_take_signal(
    direction: str, structure: StructureAnalysis
) -> tuple[bool, float, str]:
    """
    Decide if a signal should be taken based on structure.

    Returns (should_take, confidence_adjustment, reason)

    Rules:
    - If structure_bias opposes direction -> block (-1.0)
    - If near fresh order block aligned with direction -> boost (+0.15)
    - If near open FVG aligned with direction -> boost (+0.10)
    - If recent liquidity sweep aligned with direction -> boost (+0.20)
    - If no structure confluence -> neutral (0.0)
    """
    # Block if structure bias opposes direction
    if structure.structure_bias == "bearish" and direction == "long":
        return False, -1.0, "Structure bias bearish, blocking LONG"
    if structure.structure_bias == "bullish" and direction == "short":
        return False, -1.0, "Structure bias bullish, blocking SHORT"

    conf_adj = 0.0
    reasons = []

    # Fresh order block aligned with direction
    fresh_obs = [ob for ob in structure.order_blocks if ob.fresh]
    for ob in fresh_obs:
        if ob.direction == "bullish" and direction == "long":
            conf_adj += 0.15
            reasons.append(f"Fresh bullish OB at ${ob.zone_bottom:.4f}-${ob.zone_top:.4f}")
            break
        elif ob.direction == "bearish" and direction == "short":
            conf_adj += 0.15
            reasons.append(f"Fresh bearish OB at ${ob.zone_bottom:.4f}-${ob.zone_top:.4f}")
            break

    # Open FVG aligned with direction
    open_fvgs = [f for f in structure.fvgs if not f.filled]
    for fvg in open_fvgs:
        if fvg.direction == "bullish" and direction == "long":
            conf_adj += 0.10
            reasons.append(f"Open bullish FVG at ${fvg.bottom:.4f}-${fvg.top:.4f}")
            break
        elif fvg.direction == "bearish" and direction == "short":
            conf_adj += 0.10
            reasons.append(f"Open bearish FVG at ${fvg.bottom:.4f}-${fvg.top:.4f}")
            break

    # Recent liquidity sweep
    if structure.recent_sweep:
        # Sweep of sell-side liq (highs) then reversal = bearish = good for shorts
        # Sweep of buy-side liq (lows) then reversal = bullish = good for longs
        for level in structure.liquidity_levels:
            if level.level_type == "buy_side" and direction == "long":
                conf_adj += 0.20
                reasons.append("Liquidity sweep (buy-side)")
                break
            elif level.level_type == "sell_side" and direction == "short":
                conf_adj += 0.20
                reasons.append("Liquidity sweep (sell-side)")
                break

    reason = " + ".join(reasons) if reasons else "No structure confluence"
    return True, conf_adj, reason
