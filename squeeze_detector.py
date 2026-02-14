"""
=============================================================================
SQUEEZE DETECTOR — Détection avancée de compressions de volatilité
=============================================================================
Deux phases d'exploitation :
  Phase 1: Trade directionnel sur le breakout (capturer le move de 3-10%)
  Phase 2: Funding farming post-breakout (le déséquilibre crée du funding)

Méthodes de détection combinées :
  1. Bollinger Bands Width (compression relative)
  2. TTM Squeeze (BB inside Keltner Channel)
  3. ATR Compression (volatilité historique au minimum)
  4. Volume Dry-Up (volume décroissant = accumulation)
  5. Open Interest Divergence (OI monte mais prix stagne = spring chargé)

Scoring composite : chaque méthode contribue un score 0-1,
le score final détermine la "readiness" du squeeze.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class SqueezePhase(Enum):
    """Phases du cycle de volatilité."""
    NO_SQUEEZE = "no_squeeze"           # Volatilité normale
    BUILDING = "building"               # Compression en cours
    READY = "ready"                     # Squeeze mûr, breakout imminent
    FIRING = "firing"                   # Breakout en cours !
    EXPANSION = "expansion"             # Post-breakout, volatilité élevée
    COOLING = "cooling"                 # Volatilité redescend


class BreakoutDirection(Enum):
    LONG = "long"
    SHORT = "short"
    UNKNOWN = "unknown"


@dataclass
class SqueezeConfig:
    """Configuration du détecteur de squeeze."""
    
    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_percentile: float = 15.0    # Squeeze si BB width < percentile 15%
    bb_width_lookback: int = 100         # Fenêtre pour le calcul du percentile
    
    # --- Keltner Channel (pour TTM Squeeze) ---
    kc_period: int = 20
    kc_atr_mult: float = 1.5            # Multiplicateur ATR pour Keltner
    
    # --- ATR Compression ---
    atr_period: int = 14
    atr_compression_percentile: float = 20.0  # ATR < percentile 20%
    atr_lookback: int = 100
    
    # --- Volume ---
    volume_ma_period: int = 20
    volume_dry_threshold: float = 0.7    # Volume < 70% de la moyenne
    volume_dry_consecutive: int = 3      # 3 candles consécutives de volume bas
    volume_breakout_mult: float = 1.5    # Volume breakout > 1.5x moyenne
    
    # --- Scoring ---
    min_squeeze_score: float = 0.45       # Score minimum pour "BUILDING"
    ready_squeeze_score: float = 0.70     # Score minimum pour "READY"
    firing_score: float = 0.50            # Score minimum + breakout = "FIRING"
    
    # --- Direction Prediction ---
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    rsi_period: int = 14
    direction_dead_zone: float = 0.08     # Si biais directionnel < 8% => unknown

    # --- Multi-timeframe ---
    timeframes: list[str] = field(default_factory=lambda: ["1h", "4h"])

    # --- Breakout confirmation ---
    breakout_lookback_bars: int = 20
    breakout_atr_buffer: float = 0.15     # Filtre faux breakouts
    
    # --- Weights pour le score composite ---
    weight_bb: float = 0.25
    weight_ttm: float = 0.25
    weight_atr: float = 0.20
    weight_volume: float = 0.15
    weight_oi: float = 0.15


def _consecutive_true_streak(mask: pd.Series) -> pd.Series:
    """Retourne la longueur de streak consécutif de valeurs True."""
    bool_mask = mask.fillna(False).astype(bool)
    groups = (~bool_mask).cumsum()
    return bool_mask.astype(np.int32).groupby(groups).cumsum()


@dataclass
class SqueezeSignal:
    """Signal de squeeze pour un token."""
    coin: str
    timestamp: float
    phase: SqueezePhase
    score: float                          # 0-1, score composite
    direction: BreakoutDirection
    direction_confidence: float           # 0-1
    
    # Détail des composants
    bb_width_percentile: float            # Où se situe le BB width actuel
    ttm_squeeze: bool                     # BB inside Keltner ?
    ttm_squeeze_bars: int                 # Combien de bars en squeeze TTM
    atr_percentile: float                 # Où se situe l'ATR actuel
    volume_dried: bool                    # Volume sec ?
    volume_ratio: float                   # Volume actuel / moyenne
    
    # Métriques pour le trading
    atr_value: float                      # ATR absolu (pour sizing les stops)
    bb_upper: float                       # BB supérieure (résistance)
    bb_lower: float                       # BB inférieure (support)
    expected_move_pct: float              # Move attendu post-breakout
    
    # Funding info (pour Phase 2)
    current_funding: float = 0.0
    funding_predicted_direction: str = "" # "up" ou "down"
    
    def __repr__(self):
        return (
            f"SqueezeSignal({self.coin} | {self.phase.value} | "
            f"score={self.score:.2f} | dir={self.direction.value} "
            f"conf={self.direction_confidence:.1%} | "
            f"expected_move={self.expected_move_pct:.1%})"
        )
    
    @property
    def is_actionable_phase1(self) -> bool:
        """Phase 1: Trade directionnel sur breakout."""
        return (
            self.phase in (SqueezePhase.READY, SqueezePhase.FIRING)
            and self.direction != BreakoutDirection.UNKNOWN
            and self.direction_confidence >= 0.6
        )
    
    @property
    def is_actionable_phase2(self) -> bool:
        """Phase 2: Funding farming post-breakout."""
        return (
            self.phase in (SqueezePhase.EXPANSION, SqueezePhase.FIRING)
            and abs(self.current_funding) > 0.0003  # Funding > 0.03%/h
        )


# =============================================================================
# INDICATEURS TECHNIQUES
# =============================================================================

def compute_indicators(df: pd.DataFrame, config: SqueezeConfig) -> pd.DataFrame:
    """
    Calcule tous les indicateurs nécessaires.
    
    Input DataFrame doit avoir: open, high, low, close, volume
    Optionnel: oi (open interest), funding_rate
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour compute_indicators: {sorted(missing)}")

    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    prev_close = close.shift(1)

    # === Bollinger Bands ===
    df["bb_mid"] = close.rolling(config.bb_period, min_periods=config.bb_period).mean()
    df["bb_std"] = close.rolling(config.bb_period, min_periods=config.bb_period).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + config.bb_std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - config.bb_std * df["bb_std"]
    bb_span = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = bb_span / df["bb_mid"].replace(0, np.nan)
    df["bb_width_pct"] = (
        df["bb_width"]
        .rolling(config.bb_width_lookback, min_periods=config.bb_width_lookback)
        .rank(pct=True)
        * 100
    )
    df["bb_position"] = ((close - df["bb_lower"]) / bb_span).clip(-0.5, 1.5)

    # === ATR ===
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(
        alpha=1 / config.atr_period,
        adjust=False,
        min_periods=config.atr_period,
    ).mean()
    df["atr_pct"] = df["atr"] / close.replace(0, np.nan)
    df["atr_pct_rank"] = (
        df["atr_pct"]
        .rolling(config.atr_lookback, min_periods=config.atr_lookback)
        .rank(pct=True)
        * 100
    )

    # === Keltner Channel (pour TTM Squeeze) ===
    df["kc_mid"] = close.ewm(span=config.kc_period, adjust=False).mean()
    df["kc_upper"] = df["kc_mid"] + config.kc_atr_mult * df["atr"]
    df["kc_lower"] = df["kc_mid"] - config.kc_atr_mult * df["atr"]

    # TTM Squeeze: BB inside KC = squeeze actif
    df["ttm_squeeze"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    df["ttm_squeeze_bars"] = _consecutive_true_streak(df["ttm_squeeze"]).astype(int)

    # === Volume ===
    df["vol_ma"] = volume.rolling(
        config.volume_ma_period,
        min_periods=config.volume_ma_period,
    ).mean()
    df["vol_ratio"] = volume / df["vol_ma"].replace(0, np.nan)
    df["vol_dried"] = df["vol_ratio"] < config.volume_dry_threshold
    df["vol_dry_streak"] = _consecutive_true_streak(df["vol_dried"]).astype(int)
    vol_std = volume.rolling(50, min_periods=20).std(ddof=0).replace(0, np.nan)
    df["vol_zscore"] = (volume - volume.rolling(50, min_periods=20).mean()) / vol_std
    df["vol_breakout"] = (
        (df["vol_ratio"] > config.volume_breakout_mult)
        & (df["vol_zscore"].fillna(0) > 1.0)
    )

    # === EMAs / Trend ===
    df["ema_fast"] = close.ewm(span=config.ema_fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=config.ema_slow, adjust=False).mean()
    df["ema_trend"] = close.ewm(span=config.ema_trend, adjust=False).mean()
    df["ema_bullish"] = df["ema_fast"] > df["ema_slow"]
    df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]) / close.replace(0, np.nan)
    df["ema_trend_slope"] = df["ema_trend"].pct_change(5)

    # === RSI (Wilder smoothing) ===
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(
        alpha=1 / config.rsi_period,
        adjust=False,
        min_periods=config.rsi_period,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / config.rsi_period,
        adjust=False,
        min_periods=config.rsi_period,
    ).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # === Momentum ===
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(3)
    df["price_vs_mid"] = (close - df["bb_mid"]) / df["bb_mid"].replace(0, np.nan)
    df["ret_3"] = close.pct_change(3)
    df["ret_8"] = close.pct_change(8)
    df["ret_21"] = close.pct_change(21)

    # === Breakout levels ===
    prior_high = high.shift(1).rolling(
        config.breakout_lookback_bars,
        min_periods=config.breakout_lookback_bars,
    ).max()
    prior_low = low.shift(1).rolling(
        config.breakout_lookback_bars,
        min_periods=config.breakout_lookback_bars,
    ).min()
    atr_buffer = df["atr"] * config.breakout_atr_buffer
    df["breakout_up"] = close > (prior_high + atr_buffer)
    df["breakout_down"] = close < (prior_low - atr_buffer)

    # === Open Interest (si disponible) ===
    if "oi" in df.columns:
        oi = df["oi"].astype(float)
        df["oi_ma"] = oi.rolling(20, min_periods=10).mean()
        df["oi_ratio"] = oi / df["oi_ma"].replace(0, np.nan)
        df["oi_rising"] = oi.diff(5) > 0

    return df


# =============================================================================
# SCORING ENGINE
# =============================================================================

def compute_squeeze_score(row: pd.Series, config: SqueezeConfig) -> float:
    """
    Calcule un score composite de squeeze (0-1).
    Plus le score est élevé, plus le squeeze est "mûr".
    """
    scores = {}

    # 1. BB Width score (inversé : plus c'est étroit, plus le score est haut)
    bb_width_pct = row.get("bb_width_pct")
    if pd.notna(bb_width_pct):
        scores["bb"] = float(np.clip((55.0 - bb_width_pct) / 45.0, 0.0, 1.0))
    else:
        scores["bb"] = 0.0

    # 2. TTM Squeeze score avec persistance
    bars = float(row.get("ttm_squeeze_bars", 0) or 0)
    if row.get("ttm_squeeze", False):
        scores["ttm"] = float(np.clip(bars / 8.0, 0.0, 1.0))
    else:
        scores["ttm"] = float(np.clip(bars / 20.0, 0.0, 0.4))

    # 3. ATR Compression score
    atr_rank = row.get("atr_pct_rank")
    if pd.notna(atr_rank):
        scores["atr"] = float(np.clip((60.0 - atr_rank) / 50.0, 0.0, 1.0))
    else:
        scores["atr"] = 0.0

    # 4. Volume dry-up score (ratio + streak)
    vol_ratio = float(row.get("vol_ratio", 1.0) or 1.0)
    vol_streak = float(row.get("vol_dry_streak", 0) or 0)
    ratio_score = float(np.clip((1.0 - vol_ratio) / 0.5, 0.0, 1.0))
    streak_score = float(np.clip(vol_streak / max(1.0, config.volume_dry_consecutive), 0.0, 1.0))
    scores["volume"] = 0.55 * ratio_score + 0.45 * streak_score

    # 5. Open Interest score (si disponible)
    oi_ratio = row.get("oi_ratio")
    if pd.notna(oi_ratio):
        if row.get("oi_rising", False) and oi_ratio > 1.01:
            scores["oi"] = float(np.clip((oi_ratio - 1.0) * 6.0, 0.0, 1.0))
        else:
            scores["oi"] = 0.0
    else:
        scores["oi"] = None
    
    # Weighted average
    weights = {
        'bb': config.weight_bb,
        'ttm': config.weight_ttm,
        'atr': config.weight_atr,
        'volume': config.weight_volume,
        'oi': config.weight_oi,
    }
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for key, component_score in scores.items():
        if component_score is not None:
            weighted_sum += component_score * weights[key]
            total_weight += weights[key]

    if total_weight == 0:
        return 0.0

    score = weighted_sum / total_weight

    # Bonus si compression très nette et persistante
    if (
        pd.notna(bb_width_pct)
        and pd.notna(atr_rank)
        and bb_width_pct <= config.bb_width_percentile
        and atr_rank <= config.atr_compression_percentile
    ):
        score = min(1.0, score + 0.08)

    # En FIRE/EXPANSION, on dégrade le "readiness score"
    if row.get("vol_breakout", False) and pd.notna(atr_rank) and atr_rank > 65:
        score *= 0.75

    return float(np.clip(score, 0.0, 1.0))


def predict_direction(
    row: pd.Series,
    config: Optional[SqueezeConfig] = None,
) -> tuple[BreakoutDirection, float]:
    """
    Prédit la direction probable du breakout.
    Retourne (direction, confidence 0-1).
    
    Méthode multi-signal :
    - Position du prix dans les BB (haut = bullish)
    - EMA alignment
    - MACD histogram slope
    - RSI momentum
    """
    cfg = config or SqueezeConfig()

    bull_weight = 0.0
    bear_weight = 0.0
    total_weight = 0.0

    def vote(value: float, up_threshold: float, down_threshold: float, weight: float):
        nonlocal bull_weight, bear_weight, total_weight
        if not pd.notna(value) or weight <= 0:
            return
        total_weight += weight
        if value >= up_threshold:
            bull_weight += weight
        elif value <= down_threshold:
            bear_weight += weight

    # Signaux de tendance (prioritaires)
    vote(float(row.get("ema_spread", 0.0) or 0.0), 0.0015, -0.0015, 2.0)
    vote(float(row.get("ema_trend_slope", 0.0) or 0.0), 0.0020, -0.0020, 1.5)
    vote(
        float(row.get("close", np.nan)) - float(row.get("ema_trend", np.nan)),
        0.0,
        0.0,
        1.2,
    )

    # Momentum
    vote(float(row.get("macd_hist_slope", 0.0) or 0.0), 0.0, 0.0, 1.0)
    vote(float(row.get("rsi", 50.0) or 50.0), 55.0, 45.0, 0.9)
    vote(float(row.get("ret_8", 0.0) or 0.0), 0.01, -0.01, 0.9)
    vote(float(row.get("bb_position", 0.5) or 0.5), 0.56, 0.44, 0.8)

    # Breakout confirmé a un poids important
    if row.get("breakout_up", False):
        bull_weight += 2.1
        total_weight += 2.1
    elif row.get("breakout_down", False):
        bear_weight += 2.1
        total_weight += 2.1

    if total_weight <= 0:
        return BreakoutDirection.UNKNOWN, 0.0

    dominance = abs(bull_weight - bear_weight) / total_weight
    confidence = max(bull_weight, bear_weight) / total_weight

    if dominance < cfg.direction_dead_zone:
        return BreakoutDirection.UNKNOWN, confidence * 0.5

    if bull_weight > bear_weight:
        return BreakoutDirection.LONG, float(np.clip(confidence, 0.0, 0.99))
    if bear_weight > bull_weight:
        return BreakoutDirection.SHORT, float(np.clip(confidence, 0.0, 0.99))
    return BreakoutDirection.UNKNOWN, 0.0


def determine_phase(
    row: pd.Series,
    prev_phase: SqueezePhase,
    score: float,
    config: SqueezeConfig
) -> SqueezePhase:
    """
    Détermine la phase actuelle du cycle de volatilité.
    Utilise le score + les indicateurs de breakout + la phase précédente.
    """
    vol_breakout = bool(row.get("vol_breakout", False))
    bb_width_pct = row.get("bb_width_pct", 50)
    bb_pos = row.get("bb_position", 0.5)
    ttm_bars = int(row.get("ttm_squeeze_bars", 0) or 0)
    breakout_up = bool(row.get("breakout_up", False))
    breakout_down = bool(row.get("breakout_down", False))
    directional_breakout = breakout_up or breakout_down
    price_outside_bb = pd.notna(bb_pos) and (bb_pos > 0.9 or bb_pos < 0.1)

    # FIRING: breakout confirmé
    if (
        directional_breakout
        and vol_breakout
        and score >= config.firing_score * 0.85
    ):
        return SqueezePhase.FIRING

    # Transition vers FIRING plus permissive après BUILDING/READY
    if prev_phase in (SqueezePhase.BUILDING, SqueezePhase.READY):
        if vol_breakout and (directional_breakout or price_outside_bb):
            return SqueezePhase.FIRING

    # EXPANSION: volatilité relâchée après firing
    if prev_phase == SqueezePhase.FIRING:
        if pd.notna(bb_width_pct) and bb_width_pct >= 62:
            return SqueezePhase.EXPANSION
        return SqueezePhase.FIRING

    # COOLING: la vol retombe
    if prev_phase == SqueezePhase.EXPANSION:
        if pd.notna(bb_width_pct) and bb_width_pct <= 45 and not vol_breakout:
            return SqueezePhase.COOLING
        return SqueezePhase.EXPANSION

    # READY: compression forte + persistance
    if score >= config.ready_squeeze_score and ttm_bars >= 2:
        return SqueezePhase.READY

    # BUILDING: compression en cours
    if score >= config.min_squeeze_score:
        return SqueezePhase.BUILDING

    # Réamorçage depuis cooling
    if prev_phase == SqueezePhase.COOLING and score >= config.min_squeeze_score * 0.9:
        return SqueezePhase.BUILDING

    return SqueezePhase.NO_SQUEEZE


def estimate_expected_move(row: pd.Series) -> float:
    """
    Estime le mouvement de prix attendu après le breakout.
    Basé sur la compression actuelle : plus c'est compressé, plus l'explosion est grande.
    
    Méthode : ATR historique moyen vs ATR actuel.
    Le ratio donne une estimation de l'expansion à venir.
    """
    atr_pct = row.get('atr_pct', 0.01)
    atr_rank = row.get('atr_pct_rank', 50)
    
    if pd.isna(atr_pct) or pd.isna(atr_rank):
        return 0.03  # Default 3%
    
    # Plus l'ATR est compressé, plus le move attendu est grand
    # ATR au percentile 10% → expected move ~2x l'ATR moyen
    # ATR au percentile 5% → expected move ~3x l'ATR moyen
    compression_factor = max(1.0, (100 - atr_rank) / 30.0)
    
    # Le move attendu est l'ATR * facteur de compression * quelques bars
    expected_move = atr_pct * compression_factor * 3.0
    
    return min(expected_move, 0.15)  # Cap à 15%


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================

class SqueezeDetector:
    """
    Détecteur de squeeze multi-indicateur.
    
    Usage:
        detector = SqueezeDetector(config)
        
        # Avec des données OHLCV
        signal = detector.analyze(df_1h, coin="SOL", funding_rate=0.0005)
        
        if signal.is_actionable_phase1:
            # → Trade directionnel !
            
        if signal.is_actionable_phase2:
            # → Funding farming !
    """
    
    def __init__(self, config: Optional[SqueezeConfig] = None):
        self.config = config or SqueezeConfig()
        self._phase_cache: dict[str, SqueezePhase] = {}  # coin → last phase
        self._last_analysis_cache: dict[str, tuple[tuple[int, int, int], pd.Series]] = {}

    @staticmethod
    def _make_cache_key(df: pd.DataFrame) -> tuple[int, int, int]:
        """Clé stable (bars, dernier timestamp/int index, dernier close)."""
        n_rows = len(df)
        if n_rows == 0:
            return (0, 0, 0)

        try:
            close_fp = int(float(df["close"].iloc[-1]) * 1_000_000)
        except Exception:
            close_fp = 0

        last_idx = df.index[-1]
        if isinstance(last_idx, pd.Timestamp):
            return (n_rows, int(last_idx.value), close_fp)

        if "t" in df.columns:
            try:
                return (n_rows, int(df["t"].iloc[-1]), close_fp)
            except Exception:
                pass

        return (n_rows, int(n_rows), close_fp)
    
    def analyze(
        self,
        df: pd.DataFrame,
        coin: str,
        funding_rate: float = 0.0,
    ) -> SqueezeSignal:
        """
        Analyse un DataFrame OHLCV et retourne un signal de squeeze.
        
        Args:
            df: DataFrame avec colonnes open, high, low, close, volume
                Optionnel: oi (open interest)
            coin: Nom du token (ex: "SOL")
            funding_rate: Funding rate horaire actuel
        
        Returns:
            SqueezeSignal avec toutes les métriques
        """
        cache_key = self._make_cache_key(df)
        cached = self._last_analysis_cache.get(coin)
        if cached and cached[0] == cache_key:
            last = cached[1]
        else:
            # Calculer les indicateurs
            df_ind = compute_indicators(df, self.config)

            # Prendre la dernière ligne
            last = df_ind.iloc[-1].copy()
            self._last_analysis_cache[coin] = (cache_key, last)
        
        # Score composite
        score = compute_squeeze_score(last, self.config)
        
        # Direction prédite
        direction, confidence = predict_direction(last, self.config)
        
        # Phase du cycle
        prev_phase = self._phase_cache.get(coin, SqueezePhase.NO_SQUEEZE)
        phase = determine_phase(last, prev_phase, score, self.config)
        self._phase_cache[coin] = phase
        
        # Expected move
        expected_move = estimate_expected_move(last)
        
        # Prédiction funding post-breakout
        if direction == BreakoutDirection.LONG:
            funding_predicted = "up"  # Breakout haussier → longs payent → funding monte
        elif direction == BreakoutDirection.SHORT:
            funding_predicted = "down"  # Breakout baissier → shorts payent → funding négatif
        else:
            funding_predicted = "unknown"
        
        if isinstance(last.name, pd.Timestamp):
            ts = float(last.name.timestamp())
        else:
            ts = 0.0

        return SqueezeSignal(
            coin=coin,
            timestamp=ts,
            phase=phase,
            score=score,
            direction=direction,
            direction_confidence=confidence,
            bb_width_percentile=last.get('bb_width_pct', 50.0),
            ttm_squeeze=last.get('ttm_squeeze', False),
            ttm_squeeze_bars=int(last.get('ttm_squeeze_bars', 0)),
            atr_percentile=last.get('atr_pct_rank', 50.0),
            volume_dried=last.get('vol_dried', False),
            volume_ratio=last.get('vol_ratio', 1.0),
            atr_value=last.get('atr', 0.0),
            bb_upper=last.get('bb_upper', 0.0),
            bb_lower=last.get('bb_lower', 0.0),
            expected_move_pct=expected_move,
            current_funding=funding_rate,
            funding_predicted_direction=funding_predicted,
        )
    
    def scan_multiple(
        self,
        token_data: dict[str, pd.DataFrame],
        funding_rates: dict[str, float] = None,
    ) -> list[SqueezeSignal]:
        """
        Scanne plusieurs tokens et retourne les signaux triés par score.
        
        Args:
            token_data: {coin: DataFrame} pour chaque token
            funding_rates: {coin: funding_rate} optionnel
        
        Returns:
            Liste de SqueezeSignal triée par score décroissant
        """
        funding_rates = funding_rates or {}
        signals = []
        
        for coin, df in token_data.items():
            if len(df) < 100:  # Pas assez de données
                continue
            
            try:
                signal = self.analyze(
                    df, coin,
                    funding_rate=funding_rates.get(coin, 0.0),
                )
                signals.append(signal)
            except Exception as e:
                print(f"[SqueezeDetector] Error analyzing {coin}: {e}")
        
        # Trier par score décroissant
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
    
    def get_phase1_candidates(
        self,
        signals: list[SqueezeSignal],
        min_confidence: float = 0.6,
    ) -> list[SqueezeSignal]:
        """Filtre les candidats pour le trade directionnel (Phase 1)."""
        return [
            s for s in signals
            if s.phase in (SqueezePhase.READY, SqueezePhase.FIRING)
            and s.direction != BreakoutDirection.UNKNOWN
            and s.direction_confidence >= min_confidence
        ]
    
    def get_phase2_candidates(
        self,
        signals: list[SqueezeSignal],
        min_funding: float = 0.0003,
    ) -> list[SqueezeSignal]:
        """Filtre les candidats pour le funding farming (Phase 2)."""
        return [
            s for s in signals
            if s.phase in (SqueezePhase.EXPANSION, SqueezePhase.FIRING)
            and abs(s.current_funding) >= min_funding
        ]


# =============================================================================
# BACKTEST HELPER
# =============================================================================

def backtest_squeeze_signals(
    df: pd.DataFrame,
    config: SqueezeConfig = None,
    capital: float = 1000.0,
    leverage: float = 3.0,
    position_pct: float = 0.5,
    stop_atr_mult: float = 1.5,
    target_atr_mult: float = 3.0,
    trailing_stop_pct: float = 0.015,
    taker_fee: float = 0.00035,
    slippage_bps: float = 3.0,
) -> pd.DataFrame:
    """
    Backtest la Phase 1 (trades directionnels sur squeeze breakout).
    
    Returns:
        DataFrame avec les trades et métriques
    """
    config = config or SqueezeConfig()
    detector = SqueezeDetector(config)
    
    # Compute indicators une seule fois
    df_ind = compute_indicators(df, config)
    
    trades = []
    equity = capital
    in_position = False
    entry_price = 0
    entry_idx = 0
    direction = None
    stop_price = 0
    target_price = 0
    highest_since_entry = 0
    lowest_since_entry = float('inf')
    
    for i in range(100, len(df_ind)):
        row = df_ind.iloc[i]
        price = row['close']
        
        if not in_position:
            # Score et direction
            score = compute_squeeze_score(row, config)
            dir_pred, confidence = predict_direction(row, config)
            
            # Phase tracking
            coin_key = "backtest"
            prev_phase = detector._phase_cache.get(coin_key, SqueezePhase.NO_SQUEEZE)
            phase = determine_phase(row, prev_phase, score, config)
            detector._phase_cache[coin_key] = phase
            
            # Condition d'entrée : squeeze READY/FIRING + direction claire
            # OU : BUILDING avec score élevé + volume breakout
            vol_breakout = row.get('vol_breakout', False)
            expected_move = estimate_expected_move(row)
            trend_ok = (
                (dir_pred == BreakoutDirection.LONG and row.get("close", 0) > row.get("ema_trend", np.inf))
                or (dir_pred == BreakoutDirection.SHORT and row.get("close", 0) < row.get("ema_trend", -np.inf))
            )
            can_enter = (
                (phase in (SqueezePhase.READY, SqueezePhase.FIRING))
                or (phase == SqueezePhase.BUILDING and score >= 0.5 and vol_breakout)
            )
            min_conf = 0.62 if phase == SqueezePhase.READY else 0.55

            if (can_enter
                and dir_pred != BreakoutDirection.UNKNOWN
                and confidence >= min_conf
                and expected_move >= 0.02
                and trend_ok):
                
                in_position = True
                direction = dir_pred
                if direction == BreakoutDirection.LONG:
                    entry_price = price * (1 + slippage_bps / 10000)
                else:
                    entry_price = price * (1 - slippage_bps / 10000)
                entry_idx = i
                highest_since_entry = price
                lowest_since_entry = price
                
                atr = row['atr']
                if direction == BreakoutDirection.LONG:
                    stop_price = entry_price - stop_atr_mult * atr
                    target_price = entry_price + target_atr_mult * atr
                else:
                    stop_price = entry_price + stop_atr_mult * atr
                    target_price = entry_price - target_atr_mult * atr
        
        else:
            # Gestion de position
            if direction == BreakoutDirection.LONG:
                highest_since_entry = max(highest_since_entry, price)
                trailing_stop = highest_since_entry * (1 - trailing_stop_pct)
                
                hit_stop = price <= stop_price
                hit_target = price >= target_price
                hit_trailing = price <= trailing_stop and highest_since_entry > entry_price * 1.01
                max_bars = (i - entry_idx) > 50
                
            else:  # SHORT
                lowest_since_entry = min(lowest_since_entry, price)
                trailing_stop = lowest_since_entry * (1 + trailing_stop_pct)
                
                hit_stop = price >= stop_price
                hit_target = price <= target_price
                hit_trailing = price >= trailing_stop and lowest_since_entry < entry_price * 0.99
                max_bars = (i - entry_idx) > 50
            
            if hit_stop or hit_target or hit_trailing or max_bars:
                exit_price = price * (1 + slippage_bps / 10000 * (-1 if direction == BreakoutDirection.LONG else 1))
                
                if direction == BreakoutDirection.LONG:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                position_size = equity * position_pct * leverage
                gross_pnl = position_size * pnl_pct
                fees = position_size * taker_fee * 2  # Entry + exit
                net_pnl = gross_pnl - fees
                equity += net_pnl
                
                exit_reason = (
                    "stop" if hit_stop else
                    "target" if hit_target else
                    "trailing" if hit_trailing else
                    "max_bars"
                )
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': direction.value,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'net_pnl': net_pnl,
                    'equity': equity,
                    'exit_reason': exit_reason,
                    'score_at_entry': score,
                    'holding_bars': i - entry_idx,
                })
                
                in_position = False
    
    return pd.DataFrame(trades)


# =============================================================================
# DEMO / TEST
# =============================================================================

def generate_realistic_crypto_data(
    n_bars: int = 2000,
    base_price: float = 100.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère des données crypto réalistes avec des cycles de compression/expansion.
    Inclut des squeezes intentionnels pour tester le détecteur.
    """
    np.random.seed(seed)
    
    prices = [base_price]
    volumes = []
    
    # Paramètres de volatilité avec regime switching
    vol_regimes = []
    current_vol = 0.02  # 2% par bar
    
    for i in range(n_bars):
        # Regime switching : alterner compression et expansion
        cycle_pos = (i % 200) / 200  # Position dans le cycle de 200 bars
        
        if cycle_pos < 0.4:
            # Phase de compression (40% du temps)
            target_vol = 0.005 + 0.003 * np.random.random()
            current_vol = current_vol * 0.95 + target_vol * 0.05
        elif cycle_pos < 0.5:
            # Transition → explosion
            target_vol = 0.04 + 0.03 * np.random.random()
            current_vol = current_vol * 0.7 + target_vol * 0.3
        else:
            # Expansion puis retour
            target_vol = 0.015 + 0.01 * np.random.random()
            current_vol = current_vol * 0.97 + target_vol * 0.03
        
        vol_regimes.append(current_vol)
        
        # Prix
        ret = np.random.normal(0.0002, current_vol)
        # Ajouter du momentum sur les breakouts
        if cycle_pos > 0.4 and cycle_pos < 0.55:
            ret += 0.003 * np.sign(np.random.randn())
        
        prices.append(prices[-1] * (1 + ret))
        
        # Volume : inversement corrélé à la volatilité pendant compression
        base_vol = 1_000_000
        if current_vol < 0.01:
            vol = base_vol * (0.5 + 0.3 * np.random.random())  # Volume sec
        elif current_vol > 0.025:
            vol = base_vol * (1.5 + 1.0 * np.random.random())  # Volume élevé
        else:
            vol = base_vol * (0.8 + 0.4 * np.random.random())
        volumes.append(vol)
    
    prices = prices[1:]  # Remove initial price
    
    # Construire OHLCV
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes,
    })
    
    # Générer open, high, low réalistes
    df['open'] = df['close'].shift(1).fillna(base_price)
    intrabar_vol = pd.Series(vol_regimes) * 0.5
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + intrabar_vol.abs())
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - intrabar_vol.abs())
    
    # Ajouter OI simulé
    df['oi'] = 50_000_000.0
    for i in range(1, len(df)):
        oi_change = np.random.normal(0, 500_000)
        if vol_regimes[i] < 0.01:
            oi_change += 200_000  # OI monte pendant compression
        df.iloc[i, df.columns.get_loc('oi')] = max(10_000_000, df.iloc[i-1, df.columns.get_loc('oi')] + oi_change)
    
    df.index = pd.date_range('2025-01-01', periods=len(df), freq='1h')
    
    return df


def run_demo():
    """Demo complète du SqueezeDetector."""
    print("=" * 70)
    print("SQUEEZE DETECTOR — DEMO")
    print("=" * 70)
    
    # Générer des données
    df = generate_realistic_crypto_data(n_bars=2000)
    print(f"\nDonnées: {len(df)} candles 1h ({len(df)/24:.0f} jours)")
    print(f"Prix: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    
    # Initialiser le détecteur
    config = SqueezeConfig()
    detector = SqueezeDetector(config)
    
    # Scanner en temps réel (simulé)
    print("\n" + "=" * 70)
    print("SCAN EN TEMPS RÉEL (dernières 200 candles)")
    print("=" * 70)
    
    phases_history = []
    for i in range(200, len(df)):
        window = df.iloc[i-200:i+1]
        signal = detector.analyze(window, coin="SOL", funding_rate=0.0005)
        phases_history.append({
            'idx': i,
            'phase': signal.phase.value,
            'score': signal.score,
            'direction': signal.direction.value,
            'confidence': signal.direction_confidence,
        })
        
        # Afficher uniquement les transitions intéressantes
        if signal.phase in (SqueezePhase.READY, SqueezePhase.FIRING):
            print(f"  [{df.index[i]}] {signal}")
    
    # Compter les phases
    phase_counts = {}
    for p in phases_history:
        phase_counts[p['phase']] = phase_counts.get(p['phase'], 0) + 1
    
    print(f"\nDistribution des phases:")
    for phase, count in sorted(phase_counts.items()):
        pct = count / len(phases_history) * 100
        print(f"  {phase:15s}: {count:4d} ({pct:.1f}%)")
    
    # Backtest Phase 1
    print("\n" + "=" * 70)
    print("BACKTEST PHASE 1 — TRADES DIRECTIONNELS SUR BREAKOUT")
    print("=" * 70)
    
    trades_df = backtest_squeeze_signals(
        df, config,
        capital=1000.0,
        leverage=3.0,
        position_pct=0.5,
        stop_atr_mult=1.5,
        target_atr_mult=3.0,
    )
    
    if len(trades_df) == 0:
        print("  Aucun trade généré. Essayez d'ajuster les paramètres.")
        return
    
    # Métriques
    total_trades = len(trades_df)
    winners = trades_df[trades_df['net_pnl'] > 0]
    losers = trades_df[trades_df['net_pnl'] <= 0]
    win_rate = len(winners) / total_trades * 100
    
    total_pnl = trades_df['net_pnl'].sum()
    avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0
    
    profit_factor = (
        abs(winners['net_pnl'].sum() / losers['net_pnl'].sum())
        if len(losers) > 0 and losers['net_pnl'].sum() != 0 else float('inf')
    )
    
    # Max drawdown
    equity_curve = trades_df['equity']
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    final_equity = trades_df['equity'].iloc[-1]
    total_return = (final_equity - 1000) / 1000 * 100
    n_days = len(df) / 24
    monthly_return = total_return / (n_days / 30)
    
    print(f"\n  Capital initial:  $1,000")
    print(f"  Capital final:    ${final_equity:.2f}")
    print(f"  PnL total:        ${total_pnl:+.2f} ({total_return:+.1f}%)")
    print(f"  Rendement mensuel: {monthly_return:+.1f}%")
    print(f"  Nombre de trades:  {total_trades}")
    print(f"  Win rate:          {win_rate:.1f}%")
    print(f"  Avg win:           ${avg_win:+.2f}")
    print(f"  Avg loss:          ${avg_loss:+.2f}")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Max drawdown:      {max_drawdown:.1f}%")
    print(f"  Trades/jour:       {total_trades/n_days:.2f}")
    
    # Breakdown par exit reason
    print(f"\n  Exit reasons:")
    for reason, group in trades_df.groupby('exit_reason'):
        print(f"    {reason:10s}: {len(group):3d} trades, PnL ${group['net_pnl'].sum():+.2f}")
    
    # Breakdown par direction
    print(f"\n  Par direction:")
    for d, group in trades_df.groupby('direction'):
        wr = len(group[group['net_pnl'] > 0]) / len(group) * 100
        print(f"    {d:6s}: {len(group):3d} trades, WR {wr:.0f}%, PnL ${group['net_pnl'].sum():+.2f}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 — ESTIMATION FUNDING FARMING POST-BREAKOUT")
    print("=" * 70)
    
    # Estimation : après chaque breakout, le funding explose
    # On estime le funding collecté si on avait fait du delta-neutral
    estimated_funding_per_breakout = 0.0005 * 300 * 8  # 0.05%/h × $300 × 8h
    n_breakouts = len(trades_df[trades_df['exit_reason'].isin(['target', 'trailing'])])
    total_estimated_funding = n_breakouts * estimated_funding_per_breakout
    monthly_funding = total_estimated_funding / (n_days / 30)
    
    print(f"  Breakouts exploitables: {n_breakouts}")
    print(f"  Funding estimé par breakout: ${estimated_funding_per_breakout:.2f}")
    print(f"  Funding total estimé: ${total_estimated_funding:.2f}")
    print(f"  Funding mensuel estimé: ${monthly_funding:.2f}")
    
    print(f"\n  TOTAL ESTIMÉ (Phase 1 + Phase 2): ${total_pnl + total_estimated_funding:.2f}")
    print(f"  MENSUEL ESTIMÉ: ${monthly_return + monthly_funding:.2f}/mois")
    
    print("\n" + "=" * 70)
    print("PRÊT POUR INTÉGRATION AVEC TON BOT HYPERLIQUID")
    print("=" * 70)
    print("""
  Usage dans ton bot:
  
    from squeeze_detector import SqueezeDetector, SqueezeConfig
    
    detector = SqueezeDetector()
    
    # Dans ta boucle principale:
    candles = client.get_candles("SOL", "1h", limit=200)
    df = candles_to_dataframe(candles)
    funding = client.get_funding_rate("SOL")
    
    signal = detector.analyze(df, "SOL", funding_rate=funding)
    
    if signal.is_actionable_phase1:
        # PHASE 1: Trade directionnel
        if signal.direction.value == "long":
            client.place_order("SOL", is_buy=True, size=..., 
                             reduce_only=False)
        # Stop à entry - signal.atr_value * 1.5
        # Target à entry + signal.atr_value * 3.0
    
    elif signal.is_actionable_phase2:
        # PHASE 2: Funding farming delta-neutral
        # Acheter spot + short perp
        if signal.current_funding > 0:
            client.buy_spot("SOL", size=300)
            client.place_order("SOL", is_buy=False, size=300, 
                             reduce_only=False)
    """)


if __name__ == '__main__':
    run_demo()
