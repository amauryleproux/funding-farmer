"""
Patch: Ajoute l'ajustement funding rate au squeeze auto trader.
- Funding aligné (long+funding négatif OU short+funding positif) → boost confiance + taille
- Funding contre nous → réduit confiance + taille
"""

with open("squeeze_auto_trader.py", "r") as f:
    code = f.read()

# =============================================================================
# 1. Ajouter la méthode _compute_funding_adjustment dans SqueezeAutoTrader
# =============================================================================

# On l'insère juste avant _process_signals
OLD_PROCESS = '''    def _process_signals(self, signals: list[SqueezeSignal]):
        """Traite les signaux et entre en position si conditions remplies."""'''

NEW_PROCESS = '''    def _compute_funding_adjustment(self, signal) -> tuple[float, float]:
        """
        Calcule l'ajustement basé sur le funding rate.
        
        Logique:
          - LONG + funding négatif = shorts crowdés, on est payé → BOOST
          - LONG + funding positif = longs crowdés, on paye → RÉDUCTION
          - SHORT + funding positif = longs crowdés, on est payé → BOOST
          - SHORT + funding négatif = shorts crowdés, on paye → RÉDUCTION
        
        Returns:
            (confidence_adjustment, size_multiplier)
            confidence_adjustment: -0.15 à +0.15
            size_multiplier: 0.5x à 1.5x
        """
        funding = signal.current_funding  # Taux horaire
        is_long = signal.direction == BreakoutDirection.LONG
        
        # alignment > 0 = favorable (on trade contre le crowd, on est payé)
        # alignment < 0 = défavorable (on trade avec le crowd, on paye)
        alignment = -funding if is_long else funding
        
        # Normalisation: 0.0003/h (0.03%/h) = signal fort
        # Cap à ±1.0
        scale = max(-1.0, min(1.0, alignment / 0.0003))
        
        # Ajustement confiance: ±15% max
        conf_adj = scale * 0.15
        
        # Multiplicateur taille: 0.5x (très défavorable) à 1.5x (très favorable)
        size_mult = 1.0 + scale * 0.5
        
        return conf_adj, size_mult

    def _process_signals(self, signals: list[SqueezeSignal]):
        """Traite les signaux et entre en position si conditions remplies."""'''

code = code.replace(OLD_PROCESS, NEW_PROCESS)

# =============================================================================
# 2. Modifier le check de confiance pour intégrer le funding adjustment
# =============================================================================

# Remplacer le block de check confiance + l'appel à _enter_position
OLD_CONF_CHECK = '''            min_conf = self.config.min_direction_confidence
            if signal.phase == SqueezePhase.READY:
                min_conf = max(min_conf, self.config.min_ready_confidence)
            elif signal.phase == SqueezePhase.FIRING:
                min_conf = max(min_conf, self.config.min_firing_confidence)
            if signal.direction_confidence < min_conf:
                continue
            if signal.volume_ratio < self.config.min_volume_ratio:
                continue
            if signal.expected_move_pct < self.config.min_expected_move_pct:
                continue'''

NEW_CONF_CHECK = '''            # Calcul ajustement funding
            funding_conf_adj, funding_size_mult = self._compute_funding_adjustment(signal)
            adjusted_confidence = signal.direction_confidence + funding_conf_adj

            min_conf = self.config.min_direction_confidence
            if signal.phase == SqueezePhase.READY:
                min_conf = max(min_conf, self.config.min_ready_confidence)
            elif signal.phase == SqueezePhase.FIRING:
                min_conf = max(min_conf, self.config.min_firing_confidence)
            if adjusted_confidence < min_conf:
                continue
            if signal.volume_ratio < self.config.min_volume_ratio:
                continue
            if signal.expected_move_pct < self.config.min_expected_move_pct:
                continue'''

code = code.replace(OLD_CONF_CHECK, NEW_CONF_CHECK)

# =============================================================================
# 3. Passer funding_size_mult à _enter_position et ajuster l'exposition check
# =============================================================================

OLD_EXPO_ENTER = '''            # Exposition totale ?
            next_exposure = self.config.max_position_usd * self.config.leverage
            current_exposure = sum(p.size_usd for p in self.positions.values())
            if current_exposure + next_exposure > self.config.max_total_exposure_usd:
                continue

            # ✅ ENTRER
            self._enter_position(signal)'''

NEW_EXPO_ENTER = '''            # Exposition totale (ajustée par funding)
            adjusted_size = self.config.max_position_usd * funding_size_mult
            next_exposure = adjusted_size * self.config.leverage
            current_exposure = sum(p.size_usd for p in self.positions.values())
            if current_exposure + next_exposure > self.config.max_total_exposure_usd:
                continue

            # ✅ ENTRER (avec taille ajustée par funding)
            self._enter_position(signal, size_override=adjusted_size,
                                 funding_conf_adj=funding_conf_adj,
                                 funding_size_mult=funding_size_mult)'''

code = code.replace(OLD_EXPO_ENTER, NEW_EXPO_ENTER)

# =============================================================================
# 4. Modifier _enter_position pour accepter size_override et logger le funding
# =============================================================================

OLD_ENTER_SIG = '''    def _enter_position(self, signal: SqueezeSignal):
        """Ouvre une position basée sur un signal de squeeze."""
        coin = signal.coin
        is_long = signal.direction == BreakoutDirection.LONG'''

NEW_ENTER_SIG = '''    def _enter_position(self, signal: SqueezeSignal, size_override: float = 0,
                       funding_conf_adj: float = 0, funding_size_mult: float = 1.0):
        """Ouvre une position basée sur un signal de squeeze."""
        coin = signal.coin
        is_long = signal.direction == BreakoutDirection.LONG'''

code = code.replace(OLD_ENTER_SIG, NEW_ENTER_SIG)

# Modifier le calcul de size_usd pour utiliser size_override
OLD_SIZE = '''        size_usd = self.config.max_position_usd * self.config.leverage'''

NEW_SIZE = '''        base_size = size_override if size_override > 0 else self.config.max_position_usd
        size_usd = base_size * self.config.leverage'''

code = code.replace(OLD_SIZE, NEW_SIZE)

# Modifier le log pour afficher le funding adjustment
OLD_LOG_SIGNAL = '''        log.info("=" * 60)
        log.info(f"🎯 SIGNAL DÉTECTÉ — {coin}")
        log.info(f"  Phase: {signal.phase.value} | Score: {signal.score:.2f}")
        log.info(f"  Direction: {'LONG 📈' if is_long else 'SHORT 📉'} "
                 f"(conf: {signal.direction_confidence:.0%})")
        log.info(f"  Prix: {price} | ATR: {atr:.4f}")
        log.info(f"  Stop: {stop_price:.4f} | TP: {tp_price:.4f}")
        log.info(f"  Size: ${size_usd:.0f} ({self.config.leverage}x)")
        log.info(f"  Expected move: {signal.expected_move_pct:.1%}")
        log.info("=" * 60)'''

NEW_LOG_SIGNAL = '''        # Funding alignment info
        funding = signal.current_funding
        if is_long:
            funding_status = "✅ PAYÉ" if funding < 0 else "❌ PAYE" if funding > 0 else "➖ NEUTRE"
        else:
            funding_status = "✅ PAYÉ" if funding > 0 else "❌ PAYE" if funding < 0 else "➖ NEUTRE"

        log.info("=" * 60)
        log.info(f"🎯 SIGNAL DÉTECTÉ — {coin}")
        log.info(f"  Phase: {signal.phase.value} | Score: {signal.score:.2f}")
        log.info(f"  Direction: {'LONG 📈' if is_long else 'SHORT 📉'} "
                 f"(conf: {signal.direction_confidence:.0%})")
        log.info(f"  💰 Funding: {funding:+.4%}/h | {funding_status} "
                 f"| Conf adj: {funding_conf_adj:+.0%} | Size mult: {funding_size_mult:.2f}x")
        log.info(f"  Prix: {price} | ATR: {atr:.4f}")
        log.info(f"  Stop: {stop_price:.4f} | TP: {tp_price:.4f}")
        log.info(f"  Size: ${size_usd:.0f} (base ${base_size:.0f} × {self.config.leverage}x)")
        log.info(f"  Expected move: {signal.expected_move_pct:.1%}")
        log.info("=" * 60)'''

code = code.replace(OLD_LOG_SIGNAL, NEW_LOG_SIGNAL)

with open("squeeze_auto_trader.py", "w") as f:
    f.write(code)

# Vérification
checks = [
    "_compute_funding_adjustment",
    "funding_conf_adj",
    "funding_size_mult",
    "adjusted_confidence",
    "size_override",
    "PAYÉ",
    "base_size",
]
missing = [c for c in checks if c not in code]
if missing:
    print(f"⚠️ ATTENTION: éléments manquants: {missing}")
else:
    print("✅ Patch funding adjustment appliqué avec succès!")
    print()
    print("Changements:")
    print("  1. _compute_funding_adjustment() — calcule conf_adj et size_mult basés sur le funding")
    print("  2. _process_signals() — ajuste la confiance avant le check de seuil")  
    print("  3. _enter_position() — accepte size_override, log le funding alignment")
    print("  4. Logs enrichis — affiche si on est PAYÉ ou si on PAYE du funding")
    print()
    print("Logique:")
    print("  LONG + funding négatif (shorts payent) → conf +15%, size ×1.5")
    print("  LONG + funding positif (on paye)       → conf -15%, size ×0.5")
    print("  SHORT + funding positif (longs payent)  → conf +15%, size ×1.5")
    print("  SHORT + funding négatif (on paye)       → conf -15%, size ×0.5")
