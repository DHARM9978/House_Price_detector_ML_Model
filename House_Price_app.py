"""
HomeWorth - House Price Estimator  (STEP 2 : Streamlit frontend)

Loads  house_price_pipeline.pkl  (created by train_house_price_model.py).
The price is calculated ONLY when you press the "Estimate price" button.

Run :   streamlit run house_price_app.py

Files expected in the same folder as this script:
    house_price_pipeline.pkl     (required)
    housing_data.csv             (optional - only used to pick sensible "typical"
                                  values for the details that are not on the form)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = "house_price_pipeline.pkl"
DATA_FILE = "housing_data.csv"
BASE_DIR = Path(__file__).resolve().parent

# ---- Range settings (change these if you like) ----
RANGE_MULTIPLIER = 1.0   # 1.0 = estimate ± 1 RMSE (about 2 of 3 homes). 1.5 or 2.0 = wider range
ROUND_TO = 1000          # range ends are rounded to the nearest $1,000

st.set_page_config(page_title="HomeWorth · House Price Estimator",
                   page_icon="🏡", layout="wide")


# ======================================================================
#  THEME  (calm light + dark palettes)
# ======================================================================
PALETTES = {
    "light": {
        "bg1": "#F3F7F6", "bg2": "#E9F0F6", "card": "#FFFFFF", "card-soft": "#F4F8FA",
        "text": "#1F2E3C", "muted": "#68798A", "border": "#DCE5EB", "primary": "#4C9F93",
        "btn": "#3F8F84", "btn-text": "#FFFFFF", "input": "#F7FAFB",
        "ring": "rgba(76,159,147,.22)", "shadow": "0 10px 32px rgba(44,82,100,.09)",
        "hero": "linear-gradient(135deg,#3F8F84 0%,#5F86C2 100%)", "on-hero": "#FFFFFF",
        "pill": "rgba(76,159,147,.13)", "pill-text": "#2F7A70",
    },
    "dark": {
        "bg1": "#0D171C", "bg2": "#121F28", "card": "#16242C", "card-soft": "#1B2C36",
        "text": "#E6EEF1", "muted": "#8FA2AE", "border": "#26393F", "primary": "#6FC3B4",
        "btn": "#6FC3B4", "btn-text": "#0D171C", "input": "#1B2C36",
        "ring": "rgba(111,195,180,.25)", "shadow": "0 10px 32px rgba(0,0,0,.38)",
        "hero": "linear-gradient(135deg,#1F4F55 0%,#2D4A78 100%)", "on-hero": "#EAF4F6",
        "pill": "rgba(111,195,180,.14)", "pill-text": "#8AD6C8",
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { __VARS__ }

html, body, .stApp, input, textarea, button, label,
[data-testid="stMarkdownContainer"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 100%); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] *, [data-testid="stToolbar"] * { color: var(--muted); }
footer, [data-testid="stDeployButton"], [data-testid="stAppDeployButton"], .stAppDeployButton { visibility: hidden; display: none; }
.block-container { max-width: 1200px; padding: 2rem 1.6rem 9rem; }

h1, h2, h3, h4, h5, h6, .stMarkdown p, label, label p,
[data-testid="stWidgetLabel"] p { color: var(--text) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stWidgetLabel"] p { font-size: .86rem; font-weight: 500; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

/* ---------- cards ---------- */
div[class*="st-key-card_"] {
  background: var(--card); border: 1px solid var(--border); border-radius: 20px;
  padding: 1.3rem 1.4rem 1.1rem; box-shadow: var(--shadow); margin-bottom: .4rem;
}
.card-title { display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:1.05rem; color:var(--text); }
.card-sub { color: var(--muted); font-size:.84rem; margin: .15rem 0 .9rem 0; }

/* ---------- hero ---------- */
.hero-badge { display:inline-block; padding:.28rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600;
  background: var(--pill); color: var(--pill-text); letter-spacing:.02em; }
.hero-title { font-size: clamp(1.7rem, 4.2vw, 2.6rem); font-weight:700; line-height:1.15; margin:.55rem 0 .35rem; color: var(--text); }
.hero-sub { color: var(--muted); font-size: clamp(.92rem, 2vw, 1.04rem); max-width: 640px; line-height:1.5; }

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  background-color: var(--input) !important; border-color: var(--border) !important;
  border-radius: 12px !important; color: var(--text) !important; }
input, textarea { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; caret-color: var(--primary); }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--primary) !important; box-shadow: 0 0 0 3px var(--ring) !important; }
[data-baseweb="select"] * { color: var(--text) !important; }
/* newer Streamlit versions draw inputs with different elements */
[data-testid="stNumberInputContainer"], .react-aria-ComboBox > [role="group"] {
  background-color: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
[data-testid="stNumberInputContainer"]:focus-within, .react-aria-ComboBox > [role="group"]:focus-within {
  border-color: var(--primary) !important; box-shadow: 0 0 0 3px var(--ring) !important; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"], .react-aria-ComboBox button {
  background: transparent !important; color: var(--muted) !important; }
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover {
  background: var(--card-soft) !important; color: var(--text) !important; }
.react-aria-ComboBox input::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; }
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg { fill: var(--muted) !important; color: var(--muted) !important; }
[data-testid="stNumberInput"] button { background: transparent !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-testid="stNumberInput"] button:hover { background: var(--card-soft) !important; }
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"], [role="listbox"],
[data-testid="stSelectboxVirtualDropdown"] { background: var(--card) !important; border-radius: 12px; }
[data-baseweb="popover"] li, [role="option"] { background: transparent !important; color: var(--text) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"], [data-baseweb="popover"] li:hover { background: var(--card-soft) !important; }

/* ---------- buttons ---------- */
.stButton > button { width: 100%; border-radius: 999px; border: 1px solid var(--border); background: var(--card);
  color: var(--text); font-weight: 500; padding: .45rem 1rem; transition: all .15s ease; box-shadow: none; }
.stButton > button:hover { border-color: var(--primary); color: var(--primary); background: var(--card); }
.stButton > button p { color: inherit !important; }

[data-testid="stFormSubmitButton"] button { width: 100%; border-radius: 16px; border: none; background: var(--btn);
  color: var(--btn-text); font-weight: 600; font-size: 1.05rem; padding: .85rem 1rem;
  box-shadow: 0 8px 22px rgba(63,143,132,.28); transition: transform .12s ease, filter .12s ease; }
[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.07); transform: translateY(-1px); color: var(--btn-text); background: var(--btn); }
[data-testid="stFormSubmitButton"] button:active { transform: translateY(0); }
[data-testid="stFormSubmitButton"] button p { color: var(--btn-text) !important; font-weight: 600; }

/* ---------- always-visible bar: price range + Estimate button (pinned to the screen bottom) ---------- */
.st-key-action_bar { position: fixed !important; left: 50%; transform: translateX(-50%);
  bottom: calc(.9rem + env(safe-area-inset-bottom, 0px)); width: min(1120px, calc(100vw - 2rem)); z-index: 1000;
  padding: .8rem 1.1rem .9rem; border-radius: 22px; border: 1px solid var(--border); box-shadow: 0 12px 40px rgba(0,0,0,.18);
  background: color-mix(in srgb, var(--card) 94%, transparent); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
.st-key-action_bar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .9rem; align-items: center; }
.st-key-action_bar [data-testid="stColumn"] { min-width: 0 !important; }
.st-key-action_bar [data-testid="stColumn"]:first-child { flex: 1 1 0 !important; width: auto !important; min-height: 4.3rem; }
.st-key-action_bar [data-testid="stColumn"]:last-child { flex: 0 0 min(320px, 42%) !important; width: auto !important; }
.st-key-action_bar [data-testid="stElementContainer"], .st-key-action_bar [data-testid="stFormSubmitButton"],
.st-key-action_bar [data-testid="stFormSubmitButton"] > div { width: 100% !important; }
.st-key-action_bar [data-testid="stFormSubmitButton"] button { padding: .8rem .6rem; font-size: 1rem; width: 100%; }
.ab-l { font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color: var(--muted); font-weight:600; }
.ab-p { white-space: nowrap; font-size: clamp(.95rem, 4.3vw, 1.55rem); font-weight:700; color: var(--text); line-height:1.2; margin:.1rem 0; }
.ab-b { font-size:.76rem; color: var(--muted); line-height:1.35; }
.ab-b b { color: var(--primary); font-weight:600; }
.ab-empty { font-size:.92rem; color: var(--muted); margin-top:.15rem; }

/* ---------- result card ---------- */
.result-card { background: var(--hero); color: var(--on-hero); border-radius: 24px; padding: 1.7rem 1.6rem 1.4rem; box-shadow: var(--shadow); }
.result-card .eyebrow { font-size:.78rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase; opacity:.85; }
.result-card .price { font-size: clamp(1.75rem, 5.8vw, 2.7rem); font-weight:700; line-height:1.12; margin:.45rem 0 .5rem; color: var(--on-hero); }
.result-card .best { font-size:.9rem; opacity:.92; line-height:1.45; }
.result-card .best b { font-weight:600; }
.result-card .stats { display:grid; grid-template-columns: repeat(2, 1fr); gap:.6rem; margin-top:1.15rem; }
.result-card .stat { background: rgba(255,255,255,.14); border-radius:14px; padding:.65rem .85rem; }
.result-card .stat b { display:block; font-size:1rem; font-weight:600; color: var(--on-hero); }
.result-card .stat span { font-size:.74rem; opacity:.82; }
.result-card .chips { margin-top:1rem; }
.result-card .chip { display:inline-block; padding:.28rem .72rem; margin:.15rem .3rem .15rem 0; border-radius:999px;
  background: rgba(255,255,255,.17); font-size:.8rem; }
.empty-card { background: var(--card); border: 1.5px dashed var(--border); border-radius: 24px; padding: 2.2rem 1.6rem;
  text-align:center; box-shadow: var(--shadow); }
.empty-card .icon { font-size: 2.4rem; }
.empty-card .t { font-weight:600; font-size:1.1rem; color: var(--text); margin:.5rem 0 .3rem; }
.empty-card .s { color: var(--muted); font-size:.9rem; line-height:1.5; }
.note { color: var(--muted); font-size:.78rem; line-height:1.5; margin-top:.8rem; }

/* ---------- responsive ---------- */
@media (min-width: 992px) {
  .st-key-action_bar [data-testid="stColumn"]:first-child { min-height: 3.9rem; }
}
@media (max-width: 991px) {
  div[data-testid="stHorizontalBlock"]:has(.st-key-result_panel) { flex-wrap: wrap !important; }
  div[data-testid="stHorizontalBlock"]:has(.st-key-result_panel) > div[data-testid="stColumn"] {
    flex: 1 1 100% !important; min-width: 100% !important; }
}
@media (max-width: 640px) {
  .block-container { padding: 1rem .85rem 8rem; }
  div[class*="st-key-card_"] { padding: 1rem 1rem .8rem; border-radius: 16px; }
  .result-card { padding: 1.3rem 1.1rem; border-radius: 20px; }
  .st-key-action_bar { width: calc(100vw - 1rem); bottom: calc(.5rem + env(safe-area-inset-bottom, 0px));
    padding: .6rem .7rem; border-radius: 18px; }
  .st-key-action_bar [data-testid="stColumn"]:last-child { flex: 0 0 38% !important; }
  .ab-b { font-size: .7rem; }
  .ab-l { font-size: .62rem; }
  .st-key-action_bar [data-testid="stFormSubmitButton"] button { font-size: .92rem; padding: .7rem .4rem; }
}
</style>
"""


def inject_css(theme: str) -> None:
    variables = " ".join(f"--{k}: {v};" for k, v in PALETTES[theme].items())
    st.markdown(CSS.replace("__VARS__", variables), unsafe_allow_html=True)


# ======================================================================
#  STATE  (theme + last result)
# ======================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"
    try:                                           # follow the browser theme when Streamlit can tell us
        detected = st.context.theme.type
        if detected in ("light", "dark"):
            st.session_state.theme = detected
    except Exception:
        pass


def toggle_theme() -> None:
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


def reset_all() -> None:
    for k in [k for k in st.session_state if k.startswith("in_")]:
        del st.session_state[k]
    st.session_state.pop("result", None)


inject_css(st.session_state.theme)


# ======================================================================
#  LOAD MODEL  (+ read everything we need from the fitted pipeline)
# ======================================================================
def find_file(name: str):
    for folder in (BASE_DIR, Path.cwd()):
        p = folder / name
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts(path_str: str, mtime: float):
    saved = joblib.load(path_str)
    pipeline = saved["pipeline"]
    rmse = float(saved.get("rmse", float("nan")))
    pre = pipeline.named_steps["preprocess"]

    meta = {"columns": list(pre.feature_names_in_), "num_mean": {},
            "cat_options": {}, "cat_fill": {}, "cat_group": {}}
    for name, trans, cols in pre.transformers_:
        if isinstance(trans, str):          # 'drop' / 'passthrough'
            continue
        if name.startswith("num"):
            for c, m in zip(cols, trans.named_steps["scale"].mean_):
                meta["num_mean"][c] = float(m)
        elif name.startswith("cat"):
            imp, ohe = trans.named_steps["impute"], trans.named_steps["onehot"]
            for c, cats, fill in zip(cols, ohe.categories_, imp.statistics_):
                meta["cat_options"][c] = [str(x) for x in cats]
                meta["cat_fill"][c] = str(fill)
                meta["cat_group"][c] = name
    return pipeline, rmse, meta


@st.cache_data(show_spinner=False)
def load_csv_typicals(path_str: str, mtime: float):
    """Typical (median / most common) value of every column in the training csv."""
    try:
        df = pd.read_csv(path_str)
    except Exception:
        return {}
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                out[c] = float(s.median())
        else:
            # mostly-empty categorical (no pool, no alley...) -> None = "not present"
            out[c] = str(s.mode().iloc[0]) if s.notna().mean() >= 0.5 else None
    return out


model_path = find_file(MODEL_FILE)
if model_path is None:
    st.error(f"Could not find **{MODEL_FILE}**. Run `python train_house_price_model.py` first and "
             f"keep the .pkl in the same folder as this app:\n\n`{BASE_DIR}`")
    st.stop()

try:
    PIPELINE, RMSE, META = load_artifacts(str(model_path), model_path.stat().st_mtime)
except Exception as exc:
    st.error("The model file could not be loaded. Make sure it was created by "
             f"`train_house_price_model.py` and that scikit-learn versions match.\n\n`{exc}`")
    st.stop()

data_path = find_file(DATA_FILE)
CSV_TYP = load_csv_typicals(str(data_path), data_path.stat().st_mtime) if data_path else {}

COLS = META["columns"]
CAT_OPTIONS = META["cat_options"]
CAT_FILL = META["cat_fill"]
MOSTLY_ABSENT = {"Alley", "Pool QC", "Fence", "Misc Feature", "Fireplace Qu"}


def typical(col):
    """Best guess for a feature that is not on the form."""
    if col in CSV_TYP:
        v = CSV_TYP[col]
        return CAT_FILL.get(col) if v is None else v
    if col in META["num_mean"]:
        return META["num_mean"][col]
    if col in CAT_OPTIONS:
        group, fill, opts = META["cat_group"][col], CAT_FILL[col], CAT_OPTIONS[col]
        if group != "cat_na" or col in MOSTLY_ABSENT:
            return fill
        if "TA" in opts:
            return "TA"
        return next((o for o in opts if o not in ("NA", "None")), fill)
    return None


# ======================================================================
#  FRIENDLY LABELS
# ======================================================================
QUAL = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical / average", "Fa": "Fair", "Po": "Poor",
        "NA": "None / not present", "None": "None"}
OVERALL = {10: "Very excellent", 9: "Excellent", 8: "Very good", 7: "Good", 6: "Above average",
           5: "Average", 4: "Below average", 3: "Fair", 2: "Poor", 1: "Very poor"}
NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "Greens": "Greens", "GrnHill": "Green Hills", "IDOTRR": "Iowa DOT & Rail Road",
    "Landmrk": "Landmark", "MeadowV": "Meadow Village", "Mitchel": "Mitchell", "NAmes": "North Ames",
    "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of ISU", "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook", "Timber": "Timberland",
    "Veenker": "Veenker"}
ZONING = {"RL": "Residential - low density", "RM": "Residential - medium density", "FV": "Floating village",
          "RH": "Residential - high density", "C (all)": "Commercial", "A (agr)": "Agricultural",
          "I (all)": "Industrial"}
BLDG = {"1Fam": "Single-family", "2fmCon": "Two-family conversion", "Duplex": "Duplex",
        "TwnhsE": "Townhouse (end unit)", "Twnhs": "Townhouse (inside unit)", "TwnhsI": "Townhouse (inside unit)"}
STYLE = {"1Story": "1 story", "1.5Fin": "1½ story (finished)", "1.5Unf": "1½ story (unfinished)",
         "2Story": "2 story", "2.5Fin": "2½ story (finished)", "2.5Unf": "2½ story (unfinished)",
         "SFoyer": "Split foyer", "SLvl": "Split level"}
GARAGE_TYPE = {"Attchd": "Attached", "Detchd": "Detached", "BuiltIn": "Built-in", "Basment": "Basement",
               "CarPort": "Carport", "2Types": "More than one type", "NA": "No garage"}
GARAGE_FIN = {"Fin": "Finished", "RFn": "Rough finished", "Unf": "Unfinished", "NA": "No garage"}
BSMT_EXP = {"Gd": "Good exposure", "Av": "Average exposure", "Mn": "Minimum exposure", "No": "No exposure",
            "NA": "No basement"}
YESNO = {"Y": "Yes", "N": "No"}


# ======================================================================
#  INPUT HELPERS
# ======================================================================
calc_error = None
VALUES, RENDERED = {}, set()
# fields the app calculates itself from other inputs (never shown as widgets)
HIDDEN_FIELDS = {"Gr Liv Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Garage Yr Blt"}
GENERIC = {"NA": "None / not present", "None": "None"}


def num_input(box, label, col, lo, hi, step=1, fallback=0, help=None):
    if col not in COLS:
        return None
    d = typical(col)
    d = fallback if (d is None or d != d) else d
    d = int(round(min(max(d, lo), hi)))
    v = box.number_input(label, min_value=lo, max_value=hi, value=d, step=step, key=f"in_{col}", help=help)
    VALUES[col] = float(v)
    RENDERED.add(col)
    return v


def pick(box, label, col, labels=None, options=None, help=None):
    if col not in COLS:
        return None
    opts = options or CAT_OPTIONS.get(col) or []
    if not opts:
        return None
    d = typical(col)
    if isinstance(opts[0], int):
        d = int(round(float(d))) if d is not None else opts[len(opts) // 2]
        d = d if d in opts else opts[len(opts) // 2]
    elif d not in opts:
        d = opts[0]
    names = labels or {}
    v = box.selectbox(label, opts, index=opts.index(d), format_func=lambda x: names.get(x, str(x)),
                      key=f"in_{col}", help=help)
    VALUES[col] = v
    RENDERED.add(col)
    return v


def card_title(icon, title, sub):
    st.markdown(f'<div class="card-title"><span>{icon}</span>{title}</div><div class="card-sub">{sub}</div>',
                unsafe_allow_html=True)


# ======================================================================
#  CONSISTENCY RULES  (keep related fields logical, e.g. no garage -> no garage quality)
# ======================================================================
def val(col, default=0.0):
    v = VALUES.get(col, typical(col))
    return default if (v is None or v != v) else float(v)


def apply_rules():
    if "Gr Liv Area" in COLS:
        VALUES["Gr Liv Area"] = val("1st Flr SF") + val("2nd Flr SF") + val("Low Qual Fin SF")

    # garage
    if val("Garage Cars") <= 0 or val("Garage Area") <= 0:
        for c in ("Garage Type", "Garage Finish", "Garage Qual", "Garage Cond"):
            if c in COLS and META["cat_group"].get(c) == "cat_na":
                VALUES[c] = CAT_FILL[c]
        VALUES["Garage Yr Blt"] = None            # filled with the training average by the pipeline
    else:
        VALUES["Garage Yr Blt"] = VALUES.get("Year Built", typical("Year Built"))

    # basement
    total = val("Total Bsmt SF")
    if total <= 0:
        for c in ("Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"):
            if c in COLS and META["cat_group"].get(c) == "cat_na":
                VALUES[c] = CAT_FILL[c]
        VALUES.update({"BsmtFin SF 1": 0.0, "BsmtFin SF 2": 0.0, "Bsmt Unf SF": 0.0})
    else:
        t1 = float(typical("BsmtFin SF 1") or 0)
        t2 = float(typical("BsmtFin SF 2") or 0)
        t3 = float(typical("Bsmt Unf SF") or 0)
        ratio = total / (t1 + t2 + t3) if (t1 + t2 + t3) > 0 else 0
        VALUES["BsmtFin SF 1"] = t1 * ratio
        VALUES["BsmtFin SF 2"] = t2 * ratio
        VALUES["Bsmt Unf SF"] = max(total - VALUES["BsmtFin SF 1"] - VALUES["BsmtFin SF 2"], 0.0)

    # fireplace
    if val("Fireplaces") <= 0 and "Fireplace Qu" in COLS:
        VALUES["Fireplace Qu"] = CAT_FILL["Fireplace Qu"]


def build_frame() -> pd.DataFrame:
    row = {}
    for c in COLS:
        v = VALUES.get(c, typical(c))
        row[c] = np.nan if v is None else v
    df = pd.DataFrame([row], columns=COLS)
    for c in COLS:
        df[c] = df[c].astype(object) if c in CAT_OPTIONS else df[c].astype(float)
    return df


def money(x: float) -> str:
    return f"${x:,.0f}"


def bar_html(r) -> str:
    if not r:
        return ('<div class="ab-l">Estimated price range</div>'
                '<div class="ab-empty">Press Estimate price to see it</div>')
    return ('<div class="ab-l">Estimated price range</div>'
            f'<div class="ab-p">{money(r["low"])} – {money(r["high"])}</div>'
            f'<div class="ab-b">Best price for this house is around <b>{money(r["prediction"])}</b></div>')


def compute_result() -> dict:
    apply_rules()
    prediction = max(float(PIPELINE.predict(build_frame())[0]), 0.0)
    err = (RMSE if RMSE == RMSE else 0.0) * RANGE_MULTIPLIER
    low = max(round(max(prediction - err, 0.0) / ROUND_TO) * ROUND_TO, 0.0)
    high = max(round((prediction + err) / ROUND_TO) * ROUND_TO, low)
    area = val("Gr Liv Area") or (val("1st Flr SF") + val("2nd Flr SF"))

    chips = []
    if "Bedroom AbvGr" in COLS:
        chips.append(f"🛏 {int(val('Bedroom AbvGr'))} bed")
    if "Full Bath" in COLS:
        chips.append(f"🛁 {val('Full Bath') + 0.5 * val('Half Bath'):g} bath")
    if area:
        chips.append(f"📐 {area:,.0f} sq ft")
    if "Year Built" in COLS:
        chips.append(f"🏗 Built {int(val('Year Built'))}")
    if VALUES.get("Neighborhood"):
        chips.append("📍 " + NEIGHBORHOODS.get(VALUES["Neighborhood"], str(VALUES["Neighborhood"])))

    return {"prediction": prediction, "low": low, "high": high, "err": err, "area": area, "chips": chips}


# ======================================================================
#  PAGE HEADER
# ======================================================================
head_l, head_r = st.columns([5, 1.25], vertical_alignment="center")
with head_l:
    st.markdown('<div class="hero-badge">✨ Machine-learning valuation</div>'
                '<div class="hero-title">Find out what your home is worth</div>'
                '<div class="hero-sub">Fill in the details of the property, then press '
                '<b>Estimate price</b> to see the expected price range.</div>',
                unsafe_allow_html=True)
with head_r:
    label = "🌙  Dark mode" if st.session_state.theme == "light" else "☀️  Light mode"
    st.button(label, key="theme_btn", on_click=toggle_theme)

st.write("")
left, right = st.columns([3, 2], gap="large")


# ======================================================================
#  INPUT FORM  (nothing is calculated until the button is pressed)
# ======================================================================
with left:
    with st.form("estimator", border=False):
        with st.container(key="card_size"):
            card_title("📐", "Size & layout", "Floor areas and rooms")
            c1, c2, c3 = st.columns(3)
            num_input(c1, "1st floor (sq ft)", "1st Flr SF", 300, 4500, 10, 1100)
            num_input(c2, "2nd floor (sq ft, 0 if none)", "2nd Flr SF", 0, 3000, 10, 0)
            num_input(c3, "Lot size (sq ft)", "Lot Area", 1000, 100000, 100, 9500)
            c1, c2, c3, c4 = st.columns(4)
            num_input(c1, "Bedrooms", "Bedroom AbvGr", 0, 8, 1, 3)
            num_input(c2, "Full baths", "Full Bath", 0, 4, 1, 2)
            num_input(c3, "Half baths", "Half Bath", 0, 3, 1, 0)
            num_input(c4, "Total rooms", "TotRms AbvGrd", 2, 15, 1, 6, help="Rooms above ground, excluding bathrooms")

        with st.container(key="card_quality"):
            card_title("✨", "Quality & condition", "Age, finish and overall state of the home")
            c1, c2 = st.columns(2)
            pick(c1, "Overall build quality", "Overall Qual", {n: f"{n} · {OVERALL[n]}" for n in OVERALL},
                 options=list(range(1, 11)))
            pick(c2, "Overall condition", "Overall Cond", {n: f"{n} · {OVERALL[n]}" for n in range(1, 10)},
                 options=list(range(1, 10)))
            c1, c2, c3 = st.columns(3)
            pick(c1, "Kitchen quality", "Kitchen Qual", QUAL)
            pick(c2, "Exterior quality", "Exter Qual", QUAL)
            pick(c3, "Heating quality", "Heating QC", QUAL)
            c1, c2, c3 = st.columns(3)
            num_input(c1, "Year built", "Year Built", 1872, 2026, 1, 1975)
            num_input(c2, "Year remodeled", "Year Remod/Add", 1950, 2026, 1, 1995)
            pick(c3, "Central air", "Central Air", YESNO)

        with st.container(key="card_location"):
            card_title("📍", "Location & style", "Where it is and what type of home it is")
            pick(st, "Neighborhood", "Neighborhood", NEIGHBORHOODS)
            c1, c2, c3 = st.columns(3)
            pick(c1, "Zoning", "MS Zoning", ZONING)
            pick(c2, "Building type", "Bldg Type", BLDG)
            pick(c3, "House style", "House Style", STYLE)

        with st.container(key="card_extras"):
            card_title("🚗", "Basement, garage & extras", "Set sizes to 0 if the home doesn't have them")
            c1, c2, c3 = st.columns(3)
            num_input(c1, "Basement area (sq ft)", "Total Bsmt SF", 0, 6500, 10, 1000)
            pick(c2, "Basement quality", "Bsmt Qual", QUAL)
            pick(c3, "Basement exposure", "Bsmt Exposure", BSMT_EXP)
            c1, c2, c3, c4 = st.columns(4)
            num_input(c1, "Garage cars", "Garage Cars", 0, 5, 1, 2)
            num_input(c2, "Garage area (sq ft)", "Garage Area", 0, 1500, 10, 480)
            pick(c3, "Garage type", "Garage Type", GARAGE_TYPE)
            pick(c4, "Garage finish", "Garage Finish", GARAGE_FIN)
            c1, c2 = st.columns(2)
            num_input(c1, "Fireplaces", "Fireplaces", 0, 4, 1, 1)
            pick(c2, "Fireplace quality", "Fireplace Qu", QUAL, help="Ignored when there are 0 fireplaces")

        with st.container(key="card_more"):
            card_title("🔧", "More details", "Optional - these start at typical values, change only what you know")
            grid = st.columns(3)
            extra = [c for c in COLS if c not in RENDERED and c not in HIDDEN_FIELDS]
            for i, col in enumerate(extra):
                box = grid[i % 3]
                if col in CAT_OPTIONS:
                    is_quality = bool(set(CAT_OPTIONS[col]) & {"Ex", "Gd", "TA", "Fa", "Po"})
                    pick(box, col, col, QUAL if is_quality else GENERIC)
                else:
                    d = typical(col)
                    d = 0.0 if (d is None or d != d) else max(float(d), 0.0)
                    if d.is_integer():
                        v = box.number_input(col, min_value=0, value=int(d), step=1, key=f"in_{col}")
                    else:
                        v = box.number_input(col, min_value=0.0, value=round(d, 1), step=1.0,
                                             format="%.1f", key=f"in_{col}")
                    VALUES[col] = float(v)
                    RENDERED.add(col)

        # ---- always-visible bar: latest price range + the Estimate button ----
        with st.container(key="action_bar"):
            bar_price, bar_btn = st.columns([5, 3], vertical_alignment="center")
            with bar_btn:
                submitted = st.form_submit_button("Estimate price", key="estimate_btn")
            if submitted:
                try:
                    st.session_state.result = compute_result()
                except Exception as exc:
                    st.session_state.pop("result", None)
                    calc_error = str(exc)
            with bar_price:
                st.markdown(bar_html(st.session_state.get("result")), unsafe_allow_html=True)


# ======================================================================
#  RESULT PANEL
# ======================================================================
def result_html(r: dict) -> str:
    per_sqft = f"{money(r['low'] / r['area'])} – {money(r['high'] / r['area'])}" if r["area"] else "-"
    chips = "".join(f"<span class=chip>{c}</span>" for c in r["chips"])
    return (
        '<div class="result-card">'
        '<div class="eyebrow">Estimated price range</div>'
        f'<div class="price">{money(r["low"])} – {money(r["high"])}</div>'
        f'<div class="best">Best price for this house is around <b>{money(r["prediction"])}</b></div>'
        '<div class="stats">'
        f'<div class="stat"><b>{per_sqft}</b><span>Price per sq ft</span></div>'
        f'<div class="stat"><b>± {money(r["err"])}</b><span>Typical model error</span></div>'
        '</div>'
        f'<div class="chips">{chips}</div>'
        '</div>'
    )


EMPTY_HTML = ('<div class="empty-card"><div class="icon">🏡</div>'
              '<div class="t">Your estimate will appear here</div>'
              '<div class="s">Fill in the property details and press <b>Estimate price</b>.</div></div>')

with right:
    with st.container(key="result_panel"):
        if calc_error:
            st.error(f"Prediction failed: {calc_error}")
        result = st.session_state.get("result")
        if result:
            st.markdown(result_html(result), unsafe_allow_html=True)
            st.markdown('<div class="note">Changed something? Press <b>Estimate price</b> again to update. '
                        'The range is the model\'s estimate ± its typical error (RMSE), measured on homes it had '
                        'never seen. This is a statistical estimate - not a professional appraisal.</div>',
                        unsafe_allow_html=True)
            st.write("")
            st.button("↺  Start over", key="reset_btn", on_click=reset_all)
        else:
            st.markdown(EMPTY_HTML, unsafe_allow_html=True)