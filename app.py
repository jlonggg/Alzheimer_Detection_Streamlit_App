# -*- coding: utf-8 -*-
import os
import pickle as pkl
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alzheimer Risk", layout="wide")

PKL_PATH = "alzheimers_artifact.pkl"

# Fixed settings (no sidebar)
DECISION_THRESHOLD = 0.45
TOP_K = 19

# ---------------------------
# UI helpers
# ---------------------------
def stop_with_trace(title: str):
    st.error(title)
    st.code(traceback.format_exc())
    st.stop()

# ---------------------------
# Load artifact
# ---------------------------
@st.cache_resource
def load_artifact():
    with open(PKL_PATH, "rb") as f:
        return pkl.load(f)

# =========================================================
# FIX: predict_proba_compat bypass sklearn tags/version
# =========================================================
def predict_proba_compat(cal_model, X: pd.DataFrame) -> np.ndarray:
    ccs = getattr(cal_model, "calibrated_classifiers_", None)
    if not ccs:
        return cal_model.predict_proba(X)

    classes = getattr(cal_model, "classes_", np.array([0, 1]))
    n_classes = len(classes)

    n = len(X)
    proba_acc = np.zeros((n, n_classes), dtype=float)
    used = 0

    for cc in ccs:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is None:
            continue

        raw = est.predict_proba(X)
        calibrators = getattr(cc, "calibrators_", None)

        if calibrators is None:
            proba_acc += raw
            used += 1
            continue

        if n_classes == 2 and len(calibrators) == 1 and raw.shape[1] >= 2:
            p1 = calibrators[0].predict(raw[:, 1])
            p1 = np.clip(p1, 1e-15, 1 - 1e-15)
            proba_acc[:, 1] += p1
            proba_acc[:, 0] += (1 - p1)
            used += 1
        else:
            tmp = np.zeros((n, n_classes), dtype=float)
            for k in range(n_classes):
                tmp[:, k] = calibrators[k].predict(raw[:, k])
            tmp = np.clip(tmp, 1e-15, 1 - 1e-15)
            tmp = tmp / tmp.sum(axis=1, keepdims=True)
            proba_acc += tmp
            used += 1

    if used == 0:
        return cal_model.predict_proba(X)

    proba = proba_acc / used
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba

# ---------------------------
# Encoding helpers (from artifact + fallbacks)
# ---------------------------
FEATURE_TO_MAPKEY = {
    "Physical Activity Level": "map_1",
    "Depression Level": "map_1",
    "Air Pollution Exposure": "map_1",
    "Social Engagement Level": "map_1",
    "Income Level": "map_1",
    "Stress Levels": "map_1",
    "Smoking Status": "map_2",
    "Sleep Quality": "map_3",
    "Alcohol Consumption": "map_4",
    "Dietary Habits": "map_5",
    "Employment Status": "map_6",
    "Marital Status": "map_7",
}

FALLBACK_MAPS = {
    "map_1": {"Low": 0, "Medium": 1, "High": 2},
    "map_2": {"Never": 0, "Former": 1, "Current": 2},
    "map_3": {"Poor": 0, "Average": 1, "Good": 2},
    "map_4": {"Never": 0, "Occasionally": 1, "Regularly": 2},
    "map_5": {"Unhealthy": 0, "Average": 1, "Healthy": 2},
    "map_6": {"Unemployed": 0, "Retired": 1, "Employed": 2},
    "map_7": {"Single": 0, "Widowed": 1, "Married": 2},
}

def map_binary(binary_maps: dict, feature: str, label):
    m = binary_maps.get(feature)
    if not m:
        s = str(label).strip().lower()
        return 1 if s in ("yes", "true", "1", "male", "urban") else 0
    if label in m:
        return int(m[label])
    s = str(label)
    if s in m:
        return int(m[s])
    return 0

def map_ordinal(maps: dict, feature: str, label: str) -> int:
    key = FEATURE_TO_MAPKEY.get(feature)
    if not key:
        return 0
    mp = maps.get(key) or FALLBACK_MAPS.get(key, {})
    return int(mp.get(label, 0))

# ---------------------------
# Contributions (XGBoost pred_contribs)
# ---------------------------
def get_xgb_estimators(cal_model):
    ccs = getattr(cal_model, "calibrated_classifiers_", None)
    if not ccs:
        if hasattr(cal_model, "get_booster"):
            return [cal_model]
        return []
    ests = []
    for cc in ccs:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None:
            ests.append(est)
    return ests

def compute_contrib_percent(cal_model, X_one: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    import xgboost as xgb

    ests = get_xgb_estimators(cal_model)
    if len(ests) == 0:
        raise RuntimeError("Cannot extract XGBoost estimator(s) to compute contributions.")

    dm = xgb.DMatrix(X_one[features].values, feature_names=features)

    contribs_all = []
    for est in ests:
        booster = est.get_booster()
        contrib = booster.predict(dm, pred_contribs=True)[0]
        contribs_all.append(contrib)

    contrib_avg = np.mean(np.vstack(contribs_all), axis=0)
    vals = contrib_avg[:-1].astype(float)
    abs_sum = float(np.sum(np.abs(vals))) or 1.0

    dfc = pd.DataFrame(
        {
            "Feature": features,
            "ContributionLogOdds": vals,
            "ContributionPercent": (np.abs(vals) / abs_sum) * 100.0,
            "Sign": np.where(vals >= 0, 1, -1),
        }
    ).sort_values("ContributionPercent", ascending=False, kind="mergesort").reset_index(drop=True)

    return dfc

# ---------------------------
# Display helpers
# ---------------------------
def display_feature_name(f: str) -> str:
    if f.startswith("Country_"):
        return f"Continent: {f.replace('Country_', '')}"
    return f

def format_value_for_table(v):
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(round(float(v))))
    if isinstance(v, str):
        return v
    return str(v)

def pie_autopct_factory(min_pct=0.8):
    def _autopct(pct):
        return f"{pct:.1f}%" if pct >= min_pct else ""
    return _autopct

def group_for_feature(f: str) -> str:
    if f in ("Age", "Gender"):
        return "Demographics"
    if f.startswith("Country_"):
        return "Geography"
    if f in ("BMI", "Cholesterol Level"):
        return "Clinical / Lab"
    if f in ("Family History of Alzheimer’s", "Genetic Risk Factor (APOE-ε4 allele)"):
        return "Medical history"
    if f in ("Diabetes", "Hypertension"):
        return "Medical history"
    if f in (
        "Physical Activity Level",
        "Alcohol Consumption",
        "Sleep Quality",
        "Depression Level",
        "Social Engagement Level",
        "Stress Levels",
        "Urban vs Rural Living",
    ):
        return "Lifestyle & psychosocial"
    return "Other"

# ✅ FIX HERE: removed the return type annotation that crashes your pandas
def style_group_table(df: pd.DataFrame):
    def color_prog(val: str):
        if val == "Decrease risk":
            return "background-color: #ff3333; color: white; font-weight: 700;"
        if val == "Increase risk":
            return "background-color: #2ea043; color: white; font-weight: 700;"
        return ""

    styler = (
        df.style
        .applymap(color_prog, subset=["Disease Progression"])
        .set_properties(**{"text-align": "center", "vertical-align": "middle"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center"), ("vertical-align", "middle")]},
                {"selector": "td", "props": [("text-align", "center"), ("vertical-align", "middle")]},
            ]
        )
    )
    return styler

def render_styler(styler):
    try:
        st.dataframe(styler, use_container_width=True, hide_index=True)
    except Exception:
        try:
            st.dataframe(styler, use_container_width=True)
        except Exception:
            st.markdown(styler.to_html(), unsafe_allow_html=True)

# ---------------------------
# CSS (center + bigger Predict button)
# ---------------------------
st.markdown(
    """
<style>
.stButton > button {
    font-size: 20px !important;
    padding: 0.75rem 2.2rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# MAIN
# ---------------------------
if not os.path.exists(PKL_PATH):
    st.error(f"Cannot find `{PKL_PATH}` in: {os.getcwd()}")
    st.stop()

try:
    artifact = load_artifact()
    model = artifact["model"]
    FEATURES = artifact["features"]
    maps = artifact.get("maps", {})
    binary_maps = artifact.get("binary_maps", {})
except Exception:
    stop_with_trace("Failed to load PKL artifact.")

has_feat = lambda f: f in FEATURES
country_cols = [c for c in FEATURES if c.startswith("Country_")]

st.title("Alzheimer Risk Prediction")

st.subheader("Patient information")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 0, 120, 70, 1) if has_feat("Age") else 0
    bmi = st.number_input("BMI", 0, 80, 23, 1) if has_feat("BMI") else 0
    chol = st.number_input("Cholesterol Level", 0, 400, 180, 1) if has_feat("Cholesterol Level") else 0

with c2:
    fam = st.selectbox("Family History of Alzheimer’s", ["No", "Yes"]) if has_feat("Family History of Alzheimer’s") else "No"
    genetic_label = st.selectbox("Genetic Risk Factor (APOE-ε4 allele)", ["No", "Yes"]) if has_feat("Genetic Risk Factor (APOE-ε4 allele)") else "No"
    diabetes = st.selectbox("Diabetes", ["No", "Yes"]) if has_feat("Diabetes") else "No"
    hyper = st.selectbox("Hypertension", ["No", "Yes"]) if has_feat("Hypertension") else "No"
    gender = st.selectbox("Gender", ["Female", "Male"]) if has_feat("Gender") else "Female"
    urban = st.selectbox("Urban vs Rural Living", ["Rural", "Urban"]) if has_feat("Urban vs Rural Living") else "Rural"

with c3:
    low_med_high = ["Low", "Medium", "High"]
    poor_avg_good = ["Poor", "Average", "Good"]
    never_occ_reg = ["Never", "Occasionally", "Regularly"]

    pa = st.selectbox("Physical Activity Level", low_med_high) if has_feat("Physical Activity Level") else "Low"
    dep = st.selectbox("Depression Level", low_med_high) if has_feat("Depression Level") else "Low"
    social = st.selectbox("Social Engagement Level", low_med_high) if has_feat("Social Engagement Level") else "Low"
    stress = st.selectbox("Stress Levels", low_med_high) if has_feat("Stress Levels") else "Low"
    sleep = st.selectbox("Sleep Quality", poor_avg_good) if has_feat("Sleep Quality") else "Poor"
    alcohol = st.selectbox("Alcohol Consumption", never_occ_reg) if has_feat("Alcohol Consumption") else "Never"

chosen_cont = None
if len(country_cols) > 0:
    all_conts = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    chosen_cont = st.selectbox("Continent", all_conts)

b1, b2, b3 = st.columns([1, 1, 1])
with b2:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    try:
        row = {f: 0 for f in FEATURES}

        if has_feat("Age"):
            row["Age"] = float(age)
        if has_feat("BMI"):
            row["BMI"] = float(bmi)
        if has_feat("Cholesterol Level"):
            row["Cholesterol Level"] = float(chol)

        if has_feat("Family History of Alzheimer’s"):
            row["Family History of Alzheimer’s"] = map_binary(binary_maps, "Family History of Alzheimer’s", fam)
        if has_feat("Diabetes"):
            row["Diabetes"] = map_binary(binary_maps, "Diabetes", diabetes)
        if has_feat("Hypertension"):
            row["Hypertension"] = map_binary(binary_maps, "Hypertension", hyper)
        if has_feat("Gender"):
            row["Gender"] = map_binary(binary_maps, "Gender", gender)
        if has_feat("Urban vs Rural Living"):
            row["Urban vs Rural Living"] = map_binary(binary_maps, "Urban vs Rural Living", urban)

        if has_feat("Genetic Risk Factor (APOE-ε4 allele)"):
            row["Genetic Risk Factor (APOE-ε4 allele)"] = 1 if genetic_label == "Yes" else 0

        if has_feat("Physical Activity Level"):
            row["Physical Activity Level"] = map_ordinal(maps, "Physical Activity Level", pa)
        if has_feat("Depression Level"):
            row["Depression Level"] = map_ordinal(maps, "Depression Level", dep)
        if has_feat("Social Engagement Level"):
            row["Social Engagement Level"] = map_ordinal(maps, "Social Engagement Level", social)
        if has_feat("Stress Levels"):
            row["Stress Levels"] = map_ordinal(maps, "Stress Levels", stress)
        if has_feat("Sleep Quality"):
            row["Sleep Quality"] = map_ordinal(maps, "Sleep Quality", sleep)
        if has_feat("Alcohol Consumption"):
            row["Alcohol Consumption"] = map_ordinal(maps, "Alcohol Consumption", alcohol)

        if chosen_cont is not None:
            colname = f"Country_{chosen_cont}"
            if colname in row:
                row[colname] = 1

        X_one = pd.DataFrame([row], columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0)

        proba = float(predict_proba_compat(model, X_one)[0, 1])

        left, right = st.columns([1, 2])
        with left:
            st.metric("Predicted Alzheimer Risk", f"{proba*100:.2f}%")
            if proba >= DECISION_THRESHOLD:
                st.error("Risk is above the internal decision threshold.")
            else:
                st.success("Risk is below the internal decision threshold.")

        contrib = compute_contrib_percent(model, X_one, FEATURES)
        topk = min(TOP_K, len(contrib))
        contrib_top = contrib.head(topk).copy()

        display_values = {}
        if has_feat("Age"):
            display_values["Age"] = int(age)
        if has_feat("BMI"):
            display_values["BMI"] = int(bmi)
        if has_feat("Cholesterol Level"):
            display_values["Cholesterol Level"] = int(chol)

        if has_feat("Family History of Alzheimer’s"):
            display_values["Family History of Alzheimer’s"] = fam
        if has_feat("Genetic Risk Factor (APOE-ε4 allele)"):
            display_values["Genetic Risk Factor (APOE-ε4 allele)"] = genetic_label
        if has_feat("Diabetes"):
            display_values["Diabetes"] = diabetes
        if has_feat("Hypertension"):
            display_values["Hypertension"] = hyper
        if has_feat("Gender"):
            display_values["Gender"] = gender
        if has_feat("Urban vs Rural Living"):
            display_values["Urban vs Rural Living"] = urban

        if has_feat("Physical Activity Level"):
            display_values["Physical Activity Level"] = pa
        if has_feat("Depression Level"):
            display_values["Depression Level"] = dep
        if has_feat("Social Engagement Level"):
            display_values["Social Engagement Level"] = social
        if has_feat("Stress Levels"):
            display_values["Stress Levels"] = stress
        if has_feat("Sleep Quality"):
            display_values["Sleep Quality"] = sleep
        if has_feat("Alcohol Consumption"):
            display_values["Alcohol Consumption"] = alcohol

        for c in country_cols:
            display_values[c] = int(X_one.iloc[0][c])

        with right:
            st.subheader("Feature contributions (percent)")

            pie_labels = [display_feature_name(f) for f in contrib_top["Feature"].tolist()]
            pie_sizes = contrib_top["ContributionPercent"].to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                pie_sizes,
                startangle=90,
                autopct=pie_autopct_factory(min_pct=0.8),
                pctdistance=0.78,
            )
            ax.axis("equal")
            ax.legend(wedges, pie_labels, title="Features", loc="center left", bbox_to_anchor=(1.02, 0.5))
            st.pyplot(fig, clear_figure=True)

        st.divider()
        st.subheader("Feature contributions (table by group)")

        rows = []
        for _, r in contrib_top.iterrows():
            f = str(r["Feature"])
            sign = int(r["Sign"])
            prog = "Increase risk" if sign >= 0 else "Decrease risk"
            rows.append(
                {
                    "Group": group_for_feature(f),
                    "Feature": display_feature_name(f),
                    "Value": format_value_for_table(display_values.get(f, X_one.iloc[0].get(f, ""))),
                    "Disease Progression": prog,
                    "Contribution (%)": float(r["ContributionPercent"]),
                }
            )

        df_show = pd.DataFrame(rows)

        group_order = [
            "Demographics",
            "Geography",
            "Clinical / Lab",
            "Medical history",
            "Lifestyle & psychosocial",
            "Other",
        ]

        for g in group_order:
            part = df_show[df_show["Group"] == g].copy()
            if part.empty:
                continue

            part = part.sort_values("Contribution (%)", ascending=False, kind="mergesort").reset_index(drop=True)
            part.insert(0, "#", np.arange(1, len(part) + 1))
            part = part.drop(columns=["Group"])
            part["Contribution (%)"] = part["Contribution (%)"].round(2)

            st.markdown(f"## {g}")
            styler = style_group_table(part)
            render_styler(styler)

    except Exception:
        st.error("Prediction failed.")
        st.code(traceback.format_exc())