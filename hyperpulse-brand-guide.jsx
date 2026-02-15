import { useState } from "react";

const HyperPulseBrandGuide = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [logoVariant, setLogoVariant] = useState("full");

  const colors = {
    // Primary
    electricCyan: "#00F0FF",
    deepSpace: "#0A0E1A",
    midnightBlue: "#111827",
    // Accent
    pulseGreen: "#00FF88",
    warningAmber: "#FFB800",
    dangerRed: "#FF3366",
    // Neutral
    steel: "#8B95A8",
    ghost: "#1E2536",
    slate: "#2A3348",
    white: "#F0F4FF",
  };

  // SVG Logo component
  const Logo = ({ variant = "full", size = 1, light = false }) => {
    const s = size;
    const textColor = light ? colors.deepSpace : colors.white;
    const accentColor = colors.electricCyan;

    // The pulse/radar icon
    const IconMark = ({ iconSize = 40 }) => (
      <svg
        width={iconSize * s}
        height={iconSize * s}
        viewBox="0 0 40 40"
        fill="none"
      >
        {/* Outer radar ring */}
        <circle
          cx="20"
          cy="20"
          r="18"
          stroke={accentColor}
          strokeWidth="1.5"
          opacity="0.3"
        />
        {/* Middle radar ring */}
        <circle
          cx="20"
          cy="20"
          r="12"
          stroke={accentColor}
          strokeWidth="1.5"
          opacity="0.5"
        />
        {/* Inner radar ring */}
        <circle
          cx="20"
          cy="20"
          r="6"
          stroke={accentColor}
          strokeWidth="1.5"
          opacity="0.8"
        />
        {/* Center dot - the pulse */}
        <circle cx="20" cy="20" r="2.5" fill={accentColor} />
        {/* Pulse wave line going through */}
        <path
          d="M2 20 L10 20 L13 12 L16 28 L19 8 L22 32 L25 14 L28 24 L30 20 L38 20"
          stroke={accentColor}
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
        {/* Directional arrow (squeeze direction indicator) */}
        <path
          d="M32 8 L36 4 L33 9"
          stroke={colors.pulseGreen}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
    );

    if (variant === "icon") {
      return <IconMark iconSize={60} />;
    }

    if (variant === "compact") {
      return (
        <div style={{ display: "flex", alignItems: "center", gap: 8 * s }}>
          <IconMark />
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 700,
              fontSize: 20 * s,
              color: textColor,
              letterSpacing: "-0.02em",
            }}
          >
            HP
          </span>
        </div>
      );
    }

    // Full logo
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 12 * s }}>
        <IconMark />
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 700,
              fontSize: 22 * s,
              color: textColor,
              letterSpacing: "-0.02em",
              lineHeight: 1.1,
            }}
          >
            Hyper
            <span style={{ color: accentColor }}>Pulse</span>
          </span>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 400,
              fontSize: 9 * s,
              color: colors.steel,
              letterSpacing: "0.25em",
              textTransform: "uppercase",
            }}
          >
            See the squeeze
          </span>
        </div>
      </div>
    );
  };

  const ColorSwatch = ({ name, hex, usage }) => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 6,
        minWidth: 120,
      }}
    >
      <div
        style={{
          width: "100%",
          height: 64,
          backgroundColor: hex,
          borderRadius: 8,
          border:
            hex === colors.deepSpace || hex === colors.midnightBlue
              ? `1px solid ${colors.slate}`
              : "none",
          boxShadow:
            hex === colors.electricCyan || hex === colors.pulseGreen
              ? `0 4px 20px ${hex}40`
              : "none",
        }}
      />
      <div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            fontWeight: 600,
            color: colors.white,
          }}
        >
          {name}
        </div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: colors.steel,
          }}
        >
          {hex}
        </div>
        <div
          style={{
            fontFamily: "'Inter', sans-serif",
            fontSize: 11,
            color: colors.steel,
            marginTop: 2,
          }}
        >
          {usage}
        </div>
      </div>
    </div>
  );

  // Mock alert component
  const AlertPreview = () => (
    <div
      style={{
        background: `linear-gradient(135deg, ${colors.ghost} 0%, ${colors.deepSpace} 100%)`,
        border: `1px solid ${colors.slate}`,
        borderLeft: `3px solid ${colors.pulseGreen}`,
        borderRadius: 8,
        padding: 16,
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 12,
        maxWidth: 380,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 10,
        }}
      >
        <span style={{ fontSize: 14 }}>🟢</span>
        <span
          style={{ color: colors.pulseGreen, fontWeight: 700, fontSize: 13 }}
        >
          SQUEEZE ALERT
        </span>
        <span style={{ color: colors.steel }}>—</span>
        <span style={{ color: colors.white, fontWeight: 600 }}>HYPE/USDC</span>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 6,
          color: colors.steel,
          lineHeight: 1.6,
        }}
      >
        <div>
          📊 Signal:{" "}
          <span style={{ color: colors.pulseGreen, fontWeight: 600 }}>
            LONG
          </span>
        </div>
        <div>
          🎯 Conf:{" "}
          <span style={{ color: colors.electricCyan, fontWeight: 600 }}>
            82%
          </span>
        </div>
        <div>
          💰 Funding:{" "}
          <span style={{ color: colors.pulseGreen }}>-0.035%/h</span>
        </div>
        <div>
          💵 Prix: <span style={{ color: colors.white }}>$27.45</span>
        </div>
      </div>
      <div
        style={{
          marginTop: 10,
          paddingTop: 10,
          borderTop: `1px solid ${colors.slate}`,
          display: "flex",
          justifyContent: "space-between",
          color: colors.steel,
          fontSize: 10,
        }}
      >
        <span>📊 Win rate: 67% (42 signaux)</span>
        <span>⏰ il y a 2 min</span>
      </div>
    </div>
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: colors.deepSpace,
        color: colors.white,
        fontFamily: "'Inter', sans-serif",
        padding: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          background: `linear-gradient(180deg, ${colors.midnightBlue} 0%, ${colors.deepSpace} 100%)`,
          borderBottom: `1px solid ${colors.slate}`,
          padding: "32px 40px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <h1
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 28,
                fontWeight: 700,
                margin: 0,
                color: colors.white,
              }}
            >
              Hyper<span style={{ color: colors.electricCyan }}>Pulse</span>
            </h1>
            <p
              style={{
                color: colors.steel,
                fontSize: 14,
                margin: "4px 0 0 0",
                letterSpacing: "0.05em",
              }}
            >
              Brand Identity Guide — v1.0
            </p>
          </div>
          <div
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              color: colors.steel,
              background: colors.ghost,
              padding: "6px 12px",
              borderRadius: 6,
              border: `1px solid ${colors.slate}`,
            }}
          >
            hyper-pulse.xyz
          </div>
        </div>
      </div>

      <div style={{ padding: "32px 40px", maxWidth: 900 }}>
        {/* Section 1: Logo */}
        <section style={{ marginBottom: 48 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Logo
            </h2>
          </div>

          {/* Logo variants */}
          <div
            style={{
              display: "flex",
              gap: 12,
              marginBottom: 20,
              flexWrap: "wrap",
            }}
          >
            {["full", "compact", "icon"].map((v) => (
              <button
                key={v}
                onClick={() => setLogoVariant(v)}
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  padding: "6px 14px",
                  borderRadius: 6,
                  border: `1px solid ${
                    logoVariant === v ? colors.electricCyan : colors.slate
                  }`,
                  background:
                    logoVariant === v ? `${colors.electricCyan}15` : "transparent",
                  color: logoVariant === v ? colors.electricCyan : colors.steel,
                  cursor: "pointer",
                  textTransform: "capitalize",
                }}
              >
                {v}
              </button>
            ))}
          </div>

          {/* Dark background */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
            }}
          >
            <div
              style={{
                background: colors.deepSpace,
                border: `1px solid ${colors.slate}`,
                borderRadius: 12,
                padding: 32,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minHeight: 120,
              }}
            >
              <Logo variant={logoVariant} />
            </div>
            <div
              style={{
                background: "#F0F4FF",
                border: `1px solid ${colors.slate}`,
                borderRadius: 12,
                padding: 32,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minHeight: 120,
              }}
            >
              <Logo variant={logoVariant} light />
            </div>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
              marginTop: 4,
            }}
          >
            <span
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                color: colors.steel,
                textAlign: "center",
              }}
            >
              Sur fond sombre (usage principal)
            </span>
            <span
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                color: colors.steel,
                textAlign: "center",
              }}
            >
              Sur fond clair
            </span>
          </div>
        </section>

        {/* Section 2: Concept du logo */}
        <section style={{ marginBottom: 48 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Concept
            </h2>
          </div>
          <div
            style={{
              background: colors.ghost,
              border: `1px solid ${colors.slate}`,
              borderRadius: 12,
              padding: 24,
              display: "grid",
              gridTemplateColumns: "80px 1fr",
              gap: 20,
              alignItems: "start",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                paddingTop: 4,
              }}
            >
              <Logo variant="icon" size={0.8} />
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.8, color: colors.steel }}>
              <p style={{ margin: "0 0 8px 0" }}>
                L'icône combine <strong style={{ color: colors.white }}>3 éléments</strong> :
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                <span style={{ color: colors.electricCyan }}>●</span>{" "}
                <strong style={{ color: colors.white }}>Cercles radar</strong> — Surveillance continue du marché, scan 360°
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                <span style={{ color: colors.electricCyan }}>●</span>{" "}
                <strong style={{ color: colors.white }}>Ligne de pouls (ECG)</strong> — Le "pulse" du marché, la volatilité qui se compresse puis explose
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                <span style={{ color: colors.pulseGreen }}>●</span>{" "}
                <strong style={{ color: colors.white }}>Flèche directionnelle</strong> — La prédiction de direction du breakout
              </p>
              <p
                style={{
                  margin: "12px 0 0 0",
                  color: colors.white,
                  fontSize: 12,
                  fontStyle: "italic",
                }}
              >
                Le tout forme visuellement un "scope" de trader — on scanne, on détecte, on pointe.
              </p>
            </div>
          </div>
        </section>

        {/* Section 3: Colors */}
        <section style={{ marginBottom: 48 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Palette de couleurs
            </h2>
          </div>

          <p
            style={{
              fontSize: 13,
              color: colors.steel,
              marginBottom: 20,
              lineHeight: 1.6,
            }}
          >
            Dark-first. Le fond sombre évoque les terminaux de trading pro.
            Le cyan électrique attire l'œil sur les données critiques. Le vert = long/profit, le rouge = short/perte, l'ambre = attention.
          </p>

          <div style={{ marginBottom: 16 }}>
            <h3
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                color: colors.steel,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: 12,
              }}
            >
              Primaires
            </h3>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <ColorSwatch
                name="Electric Cyan"
                hex="#00F0FF"
                usage="Accent principal, CTAs, données clés"
              />
              <ColorSwatch
                name="Deep Space"
                hex="#0A0E1A"
                usage="Fond principal"
              />
              <ColorSwatch
                name="Midnight"
                hex="#111827"
                usage="Fond secondaire, cartes"
              />
            </div>
          </div>

          <div style={{ marginBottom: 16 }}>
            <h3
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                color: colors.steel,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: 12,
              }}
            >
              Sémantiques
            </h3>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <ColorSwatch
                name="Pulse Green"
                hex="#00FF88"
                usage="Long, profit, positif, bullish"
              />
              <ColorSwatch
                name="Warning Amber"
                hex="#FFB800"
                usage="Attention, confiance moyenne"
              />
              <ColorSwatch
                name="Danger Red"
                hex="#FF3366"
                usage="Short, perte, négatif, bearish"
              />
            </div>
          </div>

          <div>
            <h3
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                color: colors.steel,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                marginBottom: 12,
              }}
            >
              Neutres
            </h3>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <ColorSwatch
                name="Steel"
                hex="#8B95A8"
                usage="Texte secondaire, labels"
              />
              <ColorSwatch
                name="Ghost"
                hex="#1E2536"
                usage="Fond de cartes, sections"
              />
              <ColorSwatch
                name="Slate"
                hex="#2A3348"
                usage="Bordures, séparateurs"
              />
              <ColorSwatch
                name="Ice White"
                hex="#F0F4FF"
                usage="Texte principal sur fond sombre"
              />
            </div>
          </div>
        </section>

        {/* Section 4: Typography */}
        <section style={{ marginBottom: 48 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Typographie
            </h2>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
            }}
          >
            <div
              style={{
                background: colors.ghost,
                border: `1px solid ${colors.slate}`,
                borderRadius: 12,
                padding: 24,
              }}
            >
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  color: colors.electricCyan,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  marginBottom: 12,
                }}
              >
                Primaire — Données & UI
              </div>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 28,
                  fontWeight: 700,
                  marginBottom: 8,
                }}
              >
                JetBrains Mono
              </div>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 13,
                  color: colors.steel,
                  lineHeight: 1.6,
                }}
              >
                Monospace. Pour les données, prix, pourcentages, code, le logo, les headers. Évoque les terminaux de trading.
              </div>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  marginTop: 16,
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                }}
              >
                <span style={{ fontWeight: 700, fontSize: 16 }}>
                  Bold 700 — Titres
                </span>
                <span style={{ fontWeight: 500, fontSize: 14 }}>
                  Medium 500 — Données
                </span>
                <span
                  style={{ fontWeight: 400, fontSize: 12, color: colors.steel }}
                >
                  Regular 400 — Labels
                </span>
              </div>
            </div>

            <div
              style={{
                background: colors.ghost,
                border: `1px solid ${colors.slate}`,
                borderRadius: 12,
                padding: 24,
              }}
            >
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  color: colors.electricCyan,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  marginBottom: 12,
                }}
              >
                Secondaire — Texte & contenu
              </div>
              <div
                style={{
                  fontFamily:
                    "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
                  fontSize: 28,
                  fontWeight: 600,
                  marginBottom: 8,
                }}
              >
                Inter
              </div>
              <div
                style={{
                  fontFamily:
                    "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
                  fontSize: 13,
                  color: colors.steel,
                  lineHeight: 1.6,
                }}
              >
                Sans-serif. Pour les paragraphes, descriptions, le contenu long. Lisibilité maximale sur écran.
              </div>
              <div
                style={{
                  fontFamily:
                    "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
                  marginTop: 16,
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                }}
              >
                <span style={{ fontWeight: 600, fontSize: 16 }}>
                  SemiBold 600 — Sous-titres
                </span>
                <span style={{ fontWeight: 400, fontSize: 14 }}>
                  Regular 400 — Corps de texte
                </span>
                <span
                  style={{ fontWeight: 400, fontSize: 12, color: colors.steel }}
                >
                  Regular 400 — Petits textes
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: Alert preview */}
        <section style={{ marginBottom: 48 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Aperçu — Alerte Telegram
            </h2>
          </div>

          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <AlertPreview />
            <div
              style={{
                flex: 1,
                minWidth: 200,
                fontSize: 12,
                color: colors.steel,
                lineHeight: 1.8,
              }}
            >
              <p style={{ margin: "0 0 8px 0", color: colors.white, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                Principes des alertes :
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                → Bord gauche coloré = direction (vert=long, rouge=short)
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                → Données essentielles en haut, contexte en bas
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                → Score de confiance toujours visible en cyan
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                → Track record inclus dans chaque alerte
              </p>
              <p style={{ margin: "0 0 4px 0" }}>
                → Emojis pour scan rapide (traders scrollent vite)
              </p>
            </div>
          </div>
        </section>

        {/* Section 6: Voice & Tone */}
        <section style={{ marginBottom: 48 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Voix & Ton
            </h2>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
            }}
          >
            {[
              {
                title: "Quantitatif, pas émotionnel",
                good: '"Confiance 78% — 3/5 indicateurs alignés bearish"',
                bad: '"Ce token va DUMP !!! 🚨🚨🚨"',
              },
              {
                title: "Transparent, pas vendeur",
                good: '"Win rate 67% sur 42 signaux — voici chaque trade"',
                bad: '"99% accuracy ! Rejoins le VIP maintenant !"',
              },
              {
                title: "Concis, pas bavard",
                good: '"HYPE SHORT 78% — Funding +0.04%/h — Target $25.80"',
                bad: '"Bonjour chers traders, aujourd\'hui nous observons..."',
              },
              {
                title: "Éducatif, pas opaque",
                good: '"Le funding négatif signifie que les shorts sont crowded"',
                bad: '"Notre algorithme propriétaire a détecté..."',
              },
            ].map((item, i) => (
              <div
                key={i}
                style={{
                  background: colors.ghost,
                  border: `1px solid ${colors.slate}`,
                  borderRadius: 8,
                  padding: 16,
                }}
              >
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 12,
                    fontWeight: 600,
                    color: colors.white,
                    marginBottom: 10,
                  }}
                >
                  {item.title}
                </div>
                <div style={{ fontSize: 11, marginBottom: 6 }}>
                  <span style={{ color: colors.pulseGreen }}>✓</span>{" "}
                  <span style={{ color: colors.steel, fontStyle: "italic" }}>
                    {item.good}
                  </span>
                </div>
                <div style={{ fontSize: 11 }}>
                  <span style={{ color: colors.dangerRed }}>✗</span>{" "}
                  <span style={{ color: colors.steel, fontStyle: "italic" }}>
                    {item.bad}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Section 7: Usage rules */}
        <section style={{ marginBottom: 32 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                width: 3,
                height: 20,
                backgroundColor: colors.electricCyan,
                borderRadius: 2,
              }}
            />
            <h2
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 16,
                fontWeight: 600,
                margin: 0,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              Règles d'usage
            </h2>
          </div>

          <div
            style={{
              background: colors.ghost,
              border: `1px solid ${colors.slate}`,
              borderRadius: 12,
              padding: 24,
              fontSize: 13,
              color: colors.steel,
              lineHeight: 1.8,
            }}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <div>
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11,
                    color: colors.pulseGreen,
                    letterSpacing: "0.1em",
                    marginBottom: 8,
                  }}
                >
                  ✓ FAIRE
                </div>
                <p style={{ margin: "0 0 4px 0" }}>
                  Toujours utiliser le logo sur fond sombre
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Garder l'espace minimum autour du logo (= taille de l'icône)
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Utiliser le vert pour les signaux long / profit
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Utiliser le rouge pour les signaux short / perte
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Écrire "HyperPulse" en un mot (PascalCase)
                </p>
              </div>
              <div>
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11,
                    color: colors.dangerRed,
                    letterSpacing: "0.1em",
                    marginBottom: 8,
                  }}
                >
                  ✗ NE PAS FAIRE
                </div>
                <p style={{ margin: "0 0 4px 0" }}>
                  Changer les couleurs du logo
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Utiliser sur un fond clair sans la version light
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Mélanger les couleurs sémantiques (vert pour du bearish, etc.)
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Écrire "Hyper Pulse" ou "HYPERPULSE" ou "hyperpulse"
                </p>
                <p style={{ margin: "0 0 4px 0" }}>
                  Utiliser des polices autres que JetBrains Mono et Inter
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <div
          style={{
            borderTop: `1px solid ${colors.slate}`,
            paddingTop: 20,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Logo variant="compact" size={0.7} />
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              color: colors.steel,
            }}
          >
            See the squeeze before it fires.
          </span>
        </div>
      </div>
    </div>
  );
};

export default HyperPulseBrandGuide;
