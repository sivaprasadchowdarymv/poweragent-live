"""
Theme-neutral design system (works in Streamlit light AND dark mode): colours
are translucent over the theme background; accents use one blue + status colours.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --accent: #3B6FE0; --accent-2: #7A5AF8; --ok: #18A058; --warn: #D08700; --bad: #D64545;
  --rule: rgba(128,128,128,.22); --soft: rgba(128,128,128,.07); --tint: rgba(59,111,224,.10);
  --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; --mono: 'JetBrains Mono', ui-monospace, monospace;
  --radius: 14px;
}
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: var(--sans); }
.block-container { padding-top: 1.4rem; max-width: 1100px; }
h1, h2, h3 { letter-spacing: -0.015em; }

/* brand + hero */
.brand { display:flex; align-items:center; gap:.55rem; margin: .1rem 0 .9rem; }
.logo { width: 34px; height: 34px; border-radius: 10px; display:grid; place-items:center; color:#fff; font-weight:700;
        background: linear-gradient(135deg, var(--accent), var(--accent-2)); font-size: 1.15rem; }
.brand-title { font-weight: 700; font-size: 1.12rem; } .brand-sub { opacity:.6; font-size:.82rem; }
.hero { border-radius: var(--radius); padding: 1.6rem 1.7rem; margin-bottom: 1.1rem;
        background: linear-gradient(135deg, rgba(59,111,224,.16), rgba(122,90,248,.14)); border: 1px solid var(--rule); }
.hero-title { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: .3rem; }
.hero-sub { opacity: .8; font-size: 1rem; max-width: 62ch; }
.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:.7rem; margin: 1rem 0 .3rem; }
.step { border:1px solid var(--rule); border-radius: 12px; padding:.8rem .9rem; background: var(--soft); }
.step-n { display:inline-grid; place-items:center; width:1.5rem; height:1.5rem; border-radius:50%; font-size:.8rem;
          font-weight:700; color:#fff; background: var(--accent); margin-right:.4rem; }
.step-t { font-weight:600; } .step-d { opacity:.75; font-size:.86rem; margin-top:.25rem; }
.section-label { text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; font-weight: 600; opacity: .6; margin: .6rem 0 .35rem; }

/* cards */
.card { border: 1px solid var(--rule); border-radius: var(--radius); padding: 1rem 1.05rem; background: var(--soft);
        transition: border-color .15s ease; height: 100%; }
.card:hover { border-color: var(--accent); }
.card-title { font-weight: 600; word-break: break-all; margin-bottom: .2rem; font-size: 1rem; }
.card-sub { opacity:.7; font-size:.85rem; margin-bottom:.55rem; }
.stat-row { display:flex; flex-wrap:wrap; gap:.35rem; }
.stat { background: var(--tint); border-radius: 8px; padding: .15rem .55rem; font-size: .8rem; }
.stat b { font-variant-numeric: tabular-nums; }

/* badges, chips, pills */
.badge { display:inline-flex; align-items:center; gap:.3rem; border-radius:999px; padding:.12rem .65rem; font-size:.78rem;
         font-weight:600; border:1px solid currentColor; margin-right:.35rem; }
.badge.ok { color: var(--ok); } .badge.warn { color: var(--warn); } .badge.bad { color: var(--bad); }
.badge.info { color: var(--accent); }
.chip { display:inline-block; background: var(--tint); color: var(--accent); border-radius: 7px; padding: .1rem .5rem;
        margin: 0 .3rem .3rem 0; font-size: .78rem; font-weight: 600; }
.meta { opacity:.65; font-size:.8rem; }
.status-line { font-size: .86rem; margin: .14rem 0; }
.dot { display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.45rem; vertical-align: 1px; }
.dot.ok { background: var(--ok); } .dot.bad { background: var(--bad); } .dot.warn { background: var(--warn); }
.store { border:1px solid var(--rule); border-radius: 12px; padding:.6rem .75rem; font-size:.82rem; background: var(--soft); }

/* evidence */
.src-body { font-family: var(--mono); font-size: .78rem; white-space: pre-wrap; word-break: break-word;
            background: var(--soft); border-radius: 8px; padding: .6rem .75rem; }
.src-meta { opacity:.7; font-size:.82rem; margin-bottom:.35rem; }

/* metrics */
.metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:.6rem; margin-bottom:.8rem; }
.metric { border:1px solid var(--rule); border-radius:12px; padding:.65rem .8rem; }
.metric-name { opacity:.7; font-size:.78rem; } .metric-val { font-size:1.3rem; font-weight:700; font-variant-numeric: tabular-nums; }
.meter { background: var(--soft); height:5px; border-radius:3px; margin-top:.35rem; overflow:hidden; }
.meter > span { display:block; height:100%; border-radius:3px; }
.trace-line { font-size:.86rem; margin:.12rem 0; }

/* native widgets */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 500; }
.stChatMessage { border-radius: var(--radius); }
[data-testid="stChatInput"] textarea { font-size: .98rem; }
@media (max-width: 640px) { .hero { padding: 1.1rem; } .hero-title { font-size: 1.3rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
