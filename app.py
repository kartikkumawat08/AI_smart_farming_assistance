from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "demo_crop_data.csv"
MODEL_PATH = BASE_DIR / "crop_model.joblib"
FEATURES = ["nitrogen", "phosphorus", "potassium", "ph", "rainfall_mm", "temperature_c", "humidity_pct"]

st.set_page_config(page_title="AI Smart Farming Assistant", page_icon="🌱", layout="wide")

@st.cache_resource
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Demo dataset missing. Keep data/demo_crop_data.csv in the project folder.")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["crop"]
    # Educational demo model: the bundled dataset is synthetic, not field-validated.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced"))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    model.fit(X, y)
    return model, score, len(df)

def fertilizer_notes(n, p, k):
    notes = []
    # Broad demo thresholds only; soil-test and local recommendations take priority.
    if n < 40:
        notes.append("Nitrogen input relatively low in this demo's threshold system. Soil-test-based nitrogen planning may be needed.")
    elif n > 120:
        notes.append("Nitrogen input is high in this demo's threshold system. Avoid adding more nitrogen without a soil test.")
    else:
        notes.append("Nitrogen is within the demo's broad middle band.")
    if p < 25:
        notes.append("Phosphorus input is relatively low in this demo's threshold system; verify with a soil test.")
    elif p > 90:
        notes.append("Phosphorus input is high in this demo's threshold system; avoid extra application without expert advice.")
    else:
        notes.append("Phosphorus is within the demo's broad middle band.")
    if k < 25:
        notes.append("Potassium input is relatively low in this demo's threshold system; verify with a soil test.")
    elif k > 100:
        notes.append("Potassium input is high in this demo's threshold system; avoid extra application without expert advice.")
    else:
        notes.append("Potassium is within the demo's broad middle band.")
    return notes

def irrigation_advice(moisture, rainfall, temperature):
    if moisture < 20:
        advice = "Soil moisture is low in this prototype. Check the root zone and consider irrigation if the crop stage and local conditions support it."
    elif moisture < 40:
        advice = "Soil moisture is in the prototype's caution band. Recheck soon and consider crop stage, soil texture, and forecast."
    else:
        advice = "Soil moisture is not low by this prototype's simple threshold. Avoid watering automatically; check the crop and soil first."
    if rainfall >= 20:
        advice += " Recent/expected rainfall input is substantial; reassess before irrigating."
    if temperature >= 35:
        advice += " High temperature entered: inspect for heat stress and irrigate during suitable local hours if needed."
    return advice

try:
    model, demo_score, row_count = load_model()
except Exception as exc:
    st.error(f"Could not load project data/model: {exc}")
    st.stop()

st.title("🌱 AI Smart Farming Assistant")
st.caption("An educational AI/ML prototype for crop suggestions, soil-input checks, and irrigation awareness.")
st.warning("Prototype notice: the included crop dataset is synthetic for demonstration. Predictions and thresholds are NOT validated agronomic advice. Confirm crop choice, fertilizer, and irrigation with local agricultural experts and soil-test results.")

with st.sidebar:
    st.header("Farm inputs")
    nitrogen = st.slider("Nitrogen (N) — demo units", 0, 140, 70)
    phosphorus = st.slider("Phosphorus (P) — demo units", 0, 140, 45)
    potassium = st.slider("Potassium (K) — demo units", 0, 140, 45)
    ph = st.slider("Soil pH", 3.0, 10.0, 6.5, 0.1)
    rainfall = st.slider("Rainfall (mm; user-entered)", 0, 300, 80)
    temperature = st.slider("Temperature (°C; user-entered)", 0, 48, 27)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    moisture = st.slider("Soil moisture (%)", 0, 100, 32)
    run = st.button("Generate farming insights", type="primary", use_container_width=True)

if run:
    row = pd.DataFrame([{
        "nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium,
        "ph": ph, "rainfall_mm": rainfall, "temperature_c": temperature,
        "humidity_pct": humidity
    }])
    prediction = model.predict(row)[0]
    probs = model.predict_proba(row)[0]
    classes = model.named_steps["rf"].classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

    st.subheader("Your farming insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Demo crop suggestion", prediction.title())
    c2.metric("Model test score", f"{demo_score:.0%}", help="Held-out score on synthetic demo data only; not real-world performance.")
    c3.metric("Demo training rows", f"{row_count}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Crop suggestions")
        st.write("Top model outputs (probabilities are not real-world success chances):")
        for crop, probability in ranked:
            st.write(f"**{crop.title()}** — {probability:.0%} model score")
            st.progress(float(probability))
        st.markdown("### Soil nutrient check")
        for note in fertilizer_notes(nitrogen, phosphorus, potassium):
            st.write("• " + note)
    with right:
        st.markdown("### Irrigation awareness")
        st.info(irrigation_advice(moisture, rainfall, temperature))
        st.markdown("### pH awareness")
        if ph < 5.5:
            st.write("Entered pH is acidic relative to many common crop ranges. Verify soil-test results and crop-specific needs.")
        elif ph > 8.0:
            st.write("Entered pH is alkaline relative to many common crop ranges. Verify soil-test results and crop-specific needs.")
        else:
            st.write("Entered pH falls in a broad middle band for this demo. Crop-specific suitability still varies.")
        st.markdown("### Before taking action")
        st.write("1. Check a current soil test and crop growth stage.")
        st.write("2. Review local weather and water availability.")
        st.write("3. Confirm with a Krishi Vigyan Kendra (KVK), agriculture officer, or qualified agronomist.")

    st.download_button(
        "Download this input summary (CSV)",
        data=row.to_csv(index=False).encode("utf-8"),
        file_name="farm_input_summary.csv",
        mime="text/csv"
    )
else:
    st.info("Sidebar mein farm values set karke **Generate farming insights** click karo.")
    st.markdown("### What this prototype demonstrates")
    st.markdown("- Machine-learning classification for a demo crop suggestion\n- Rule-based soil nutrient and irrigation awareness\n- Explainable limitations and responsible-use warning")
st.divider()
st.caption("Built for learning and internship demonstration. No live weather API, sensor connection, or field-validated recommendation engine is included.")
