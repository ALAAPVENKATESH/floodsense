"""Single stylesheet for FloodSense. Tokens are filled in per theme mode."""

from __future__ import annotations

import streamlit as st

from app.ui.theme import TOKENS

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, [class*="css"], button, input, textarea, select, label, p, h1, h2, h3, span, div {
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}
.stApp, [data-testid="stAppViewContainer"] {
  background: [[bg]] !important;
  color: [[text]] !important;
}
[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container {
  padding-top: 1.6rem !important;
  padding-bottom: 4.75rem !important;
  max-width: 1180px;
  animation: fs-in 0.35s ease;
}
@keyframes fs-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: none; }
}
[data-testid="stSidebar"] {
  background: [[sidebar]] !important;
  border-right: 1px solid [[border]];
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
[data-testid="stSidebarUserContent"] { padding-bottom: 3.5rem; }
h1, h2, h3, p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: [[text]]; }
[data-testid="stCaptionContainer"] p { color: [[muted]] !important; font-size: 12px !important; }
a { color: [[brand]]; }

[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: [[muted]] !important;
  text-align: left !important;
  justify-content: flex-start !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.75rem !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
  background: [[surface2]] !important;
  color: [[text]] !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
  background: [[brand]] !important;
  color: #fff !important;
  border: none !important;
  text-align: left !important;
  justify-content: flex-start !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}
button[data-testid="stBaseButton-primary"] {
  background: [[brand]] !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: background 0.15s ease, transform 0.15s ease;
}
button[data-testid="stBaseButton-primary"]:hover { background: [[brandHover]] !important; }
button[data-testid="stBaseButton-secondary"] {
  background: [[surface]] !important;
  color: [[text]] !important;
  border: 1px solid [[border]] !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  border: none !important;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: [[brand]] !important; }

[data-testid="stNumberInput"] input, [data-baseweb="input"] > div, [data-testid="stTextInput"] input {
  background: [[surface]] !important;
  color: [[text]] !important;
  border-color: [[border]] !important;
  border-radius: 10px !important;
}
[data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label, [data-testid="stRadio"] label {
  font-size: 13px !important;
  font-weight: 500 !important;
}
[data-baseweb="select"] > div {
  background: [[surface]] !important;
  border-color: [[border]] !important;
  border-radius: 10px !important;
  color: [[text]] !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: [[brand]] !important; }
[data-testid="stFileUploader"] section {
  background: [[surface]] !important;
  border: 1px dashed [[border]] !important;
  border-radius: 14px !important;
  color: [[text]] !important;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: [[muted]] !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
  background: [[surface]] !important;
  border: 1px solid [[border]] !important;
  border-radius: 16px !important;
  box-shadow: [[shadow]];
  padding: 4px 8px 12px !important;
}
[data-baseweb="tab-highlight"] { background: [[brand]] !important; }
[data-baseweb="tab"] { color: [[muted]] !important; }
[data-baseweb="tab"][aria-selected="true"] { color: [[text]] !important; }
[data-testid="stProgress"] > div > div > div { background: [[brand]] !important; }
[data-testid="stButtonGroup"] button { border-radius: 8px !important; }
[data-testid="stRadio"] label { color: [[text]] !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: [[scroll]]; border-radius: 99px; }
::selection { background: [[selection]]; }

@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"] { flex-direction: column; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
  .fs-page-title { font-size: 26px !important; }
  .fs-page-head, .fs-hero { flex-direction: column; align-items: flex-start; }
}

.fs-brand { display: flex; gap: 12px; align-items: center; margin: 0 0 18px 4px; }
.fs-brand-name { font-size: 16px; font-weight: 600; letter-spacing: -0.02em; color: [[text]]; line-height: 1.1; }
.fs-brand-tag { font-size: 12px; color: [[muted]]; margin-top: 2px; }
.fs-nav-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: [[muted]]; margin: 14px 4px 6px; }
.fs-hr { border: none; border-top: 1px solid [[border]]; margin: 14px 0; }
.fs-side-meta { font-size: 12px; color: [[muted]]; line-height: 1.45; margin: 8px 4px 0; }

.fs-page-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; margin-bottom: 16px; }
.fs-page-title { font-size: 30px; font-weight: 600; letter-spacing: -0.03em; margin: 0; color: [[text]]; line-height: 1.15; }
.fs-page-desc { font-size: 14px; color: [[muted]]; margin: 6px 0 0; }
.fs-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.fs-chip { display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 500; border: 1px solid [[border]]; background: [[surface]]; color: [[text]]; }
.fs-chip-warn { background: [[bannerBg]]; color: [[bannerText]]; border-color: [[bannerBorder]]; }
.fs-chip-good { background: [[lowBg]]; color: #16A34A; border-color: transparent; }
.fs-chip-neutral { background: [[surface2]]; color: [[text]]; }

.fs-banner { display: flex; gap: 12px; align-items: flex-start; background: [[bannerBg]]; color: [[bannerText]]; border: 1px solid [[bannerBorder]]; border-radius: 12px; padding: 12px 14px; margin-bottom: 16px; }
.fs-banner > div:first-child { flex: 1; min-width: 0; }
.fs-banner strong { display: block; font-size: 13px; font-weight: 600; }
.fs-banner span { display: block; font-size: 13px; line-height: 1.45; white-space: normal; }
.fs-banner-x { margin-left: auto; font-size: 12px; opacity: 0.7; white-space: nowrap; }

.fs-hero { background: linear-gradient(180deg, [[surface]] 0%, [[surface2]] 100%); border: 1px solid [[border]]; border-radius: 16px; box-shadow: [[shadow]]; padding: 22px 24px; }
.fs-kicker { font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: [[brand]]; }
.fs-hero h2 { font-size: 28px; font-weight: 600; letter-spacing: -0.03em; margin: 6px 0 8px; color: [[text]]; }
.fs-hero p { margin: 0; font-size: 14px; color: [[muted]]; max-width: 62ch; }

.fs-kpi, .fs-card, .fs-empty, .fs-risk, .fs-callout {
  background: [[surface]];
  border: 1px solid [[border]];
  border-radius: 16px;
  box-shadow: [[shadow]];
}
.fs-kpi { padding: 18px 20px; min-height: 118px; transition: transform 0.15s ease, box-shadow 0.15s ease; }
.fs-kpi:hover { transform: translateY(-2px); box-shadow: [[shadowHover]]; }
.fs-kpi-label { font-size: 12px; font-weight: 600; color: [[muted]]; letter-spacing: 0.02em; }
.fs-kpi-value { font-size: 28px; font-weight: 600; letter-spacing: -0.03em; margin: 8px 0 4px; color: [[text]]; }
.fs-kpi-delta { font-size: 12px; color: [[muted]]; }
.fs-card { padding: 20px 22px; }
.fs-section { font-size: 18px; font-weight: 600; margin: 0 0 4px; color: [[text]]; }
.fs-section-note { font-size: 13px; color: [[muted]]; margin: 0 0 12px; }

.fs-risk { padding: 22px 24px; }
.fs-risk-kicker { font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: [[muted]]; }
.fs-risk-row { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
.fs-risk-class { font-size: 40px; font-weight: 600; letter-spacing: -0.04em; line-height: 1; }
.fs-badge { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 700; letter-spacing: 0.04em; }
.fs-badge-LOW { background: [[lowBg]]; color: #16A34A; }
.fs-badge-MODERATE { background: [[modBg]]; color: #D97706; }
.fs-badge-HIGH { background: [[highBg]]; color: #DC2626; }
.fs-risk-sub { margin-top: 8px; font-size: 14px; color: [[muted]]; }

.fs-empty { padding: 28px 22px; text-align: center; }
.fs-empty-title { font-size: 16px; font-weight: 600; color: [[text]]; }
.fs-empty p { font-size: 13px; color: [[muted]]; margin: 6px 0 0; }
.fs-alert { background: [[bannerBg]]; color: [[bannerText]]; border: 1px solid [[bannerBorder]]; border-radius: 10px; padding: 8px 10px; font-size: 13px; margin: 4px 0 8px; }
.fs-callout { padding: 12px 14px; font-size: 13px; color: [[muted]]; border-left: 3px solid [[brand]]; }
.fs-pill { display: inline-flex; margin: 0 8px 8px 0; padding: 6px 10px; border-radius: 999px; background: [[surface2]]; color: [[text]]; border: 1px solid [[border]]; font-size: 12px; font-weight: 500; }
.fs-pipeline { display: flex; flex-wrap: wrap; gap: 8px; list-style: none; padding: 0; margin: 0; }
.fs-pipeline li { background: [[surface2]]; color: [[text]]; border: 1px solid [[border]]; border-radius: 999px; padding: 6px 12px; font-size: 13px; font-weight: 500; }
.fs-node { border-left: 2px solid [[border]]; padding: 6px 10px; margin: 4px 0; font-size: 13px; color: [[text]]; background: [[surface2]]; border-radius: 0 10px 10px 0; }
.fs-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.fs-table th { text-align: left; font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: [[muted]]; font-weight: 600; padding: 8px 10px; border-bottom: 1px solid [[border]]; }
.fs-table td { padding: 8px 10px; border-bottom: 1px solid [[border]]; color: [[text]]; vertical-align: middle; }
.fs-footer {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 100; height: 36px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; color: [[muted]];
  background: [[bg]]; border-top: 1px solid [[border]];
}
.fs-group { font-size: 13px; font-weight: 600; color: [[text]]; margin: 10px 0 4px; }
"""


def inject_styles(mode: str) -> None:
    """Inject the stylesheet for the active light or dark mode."""
    tokens = TOKENS["dark" if mode == "dark" else "light"]
    css = _CSS
    for key, value in tokens.items():
        css = css.replace(f"[[{key}]]", value)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
