import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import google.generativeai as genai
import json
from pydantic import BaseModel, Field

# Page Configuration
st.set_page_config(
    page_title="Water Quality Assessment AI",
    page_icon="💧",
    layout="wide"
)

# Professional Creative Water Theme CSS
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #f0f4f8 !important;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #90e0ef 0%, #00b4d8 100%);
        color: #03045e;
        box-shadow: 0 6px 20px rgba(144, 224, 239, 0.6);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.title("💧 AI-Powered Water Quality Assessment System")
st.markdown("##### Advanced Water Safety Analytics powered by **Google Gemini AI** & **Fuzzy Logic Control Systems**")
st.markdown("---")

# Retrieve Gemini API Key securely from Streamlit Secrets
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

with st.sidebar:
    st.image("https://img.icons8.com/color/96/water.png", width=70)
    st.header("🌊 Dashboard Info")
    st.info("This professional tool evaluates water potability using artificial intelligence text parsing combined with precise fuzzy logic mathematics seamlessly.")
    st.markdown("---")
    st.markdown("### 🫧 Water Vibe Active")
    st.write("System running on secure cloud configuration.")

# Fuzzy Logic Calculation Engine
def evaluate_fuzzy_water(pH_val, turbidity_val, tds_val):
    pH = ctrl.Antecedent(np.arange(0, 14.1, 0.1), 'pH')
    turbidity = ctrl.Antecedent(np.arange(0, 51, 1), 'turbidity')
    tds = ctrl.Antecedent(np.arange(0, 1001, 1), 'tds')
    quality = ctrl.Consequent(np.arange(0, 101, 1), 'quality')

    pH['acidic'] = fuzz.trapmf(pH.universe, [0, 0, 6.5, 6.8])
    pH['safe'] = fuzz.trimf(pH.universe, [6.5, 7.2, 8.5])
    pH['alkaline'] = fuzz.trapmf(pH.universe, [8.0, 8.5, 14, 14])

    turbidity['clear'] = fuzz.trapmf(turbidity.universe, [0, 0, 4, 6])
    turbidity['moderate'] = fuzz.trimf(turbidity.universe, [5, 10, 20])
    turbidity['murky'] = fuzz.trapmf(turbidity.universe, [15, 25, 50, 50])

    tds['low'] = fuzz.trapmf(tds.universe, [0, 0, 150, 300])
    tds['moderate'] = fuzz.trimf(tds.universe, [250, 500, 750])
    tds['high'] = fuzz.trapmf(tds.universe, [700, 850, 1000, 1000])

    quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 40])
    quality['moderate'] = fuzz.trimf(quality.universe, [30, 50, 70])
    quality['good'] = fuzz.trimf(quality.universe, [60, 100, 100])

    rule1 = ctrl.Rule(pH['safe'] & turbidity['clear'] & tds['low'], quality['good'])
    rule2 = ctrl.Rule(pH['acidic'] | turbidity['murky'] | tds['high'], quality['poor'])
    rule3 = ctrl.Rule(turbidity['moderate'] | tds['moderate'], quality['moderate'])
    rule4 = ctrl.Rule(pH['alkaline'], quality['moderate'])

    quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    water_sim = ctrl.ControlSystemSimulation(quality_ctrl)

    water_sim.input['pH'] = float(np.clip(pH_val, 0, 14))
    water_sim.input['turbidity'] = float(np.clip(turbidity_val, 0, 50))
    water_sim.input['tds'] = float(np.clip(tds_val, 0, 1000))

    try:
        water_sim.compute()
        return water_sim.output['quality']
    except Exception:
        return 50.0

# Input Selection Tabs (AI Text Analysis vs Manual Sliders)
tab1, tab2 = st.tabs(["🤖 AI Text Description Analysis", "🎛️ Manual Parameter Sliders"])

ph_val, turbidity_val, tds_val = 7.0, 5.0, 300.0

with tab1:
    st.subheader("Natural Language Water Condition Input")
    user_query = st.text_area(
        "Describe water sample condition in plain text:",
        "The water looks a bit cloudy, tastes slightly metallic, and the pH is around 6.8 with TDS near 350 ppm.",
        height=110
    )
    if st.button("🚀 Analyze via Gemini AI"):
        if api_key and user_query:
            try:
                with st.spinner("✨ Gemini AI is extracting parameters..."):
                    genai.configure(api_key=api_key)
                    # Using the standard gemini-1.5-flash with proper JSON prompt instruction
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""Extract water test parameters from the following text and return ONLY a valid JSON object with keys: "pH" (float), "turbidity" (float), and "tds" (float). No markdown formatting or extra text.
                    Text: {user_query}"""
                    
                    response = model.generate_content(prompt)
                    clean_text = response.text.strip().replace("```json", "").replace("```", "")
                    data = json.loads(clean_text)
                    
                    ph_val = float(data.get("pH", 7.0))
                    turbidity_val = float(data.get("turbidity", 5.0))
                    tds_val = float(data.get("tds", 300.0))
                    st.success("Parameters successfully extracted via Gemini AI!")
            except Exception as e:
                st.warning(f"AI Extraction warning ({e}). Using default fallback parameters.")
        elif not api_key:
            st.warning("⚠️ Gemini API key missing in Streamlit Secrets configuration.")

with tab2:
    st.subheader("Manual Parameter Adjustment")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        ph_val = st.slider("pH Level", 0.0, 14.0, 7.2, 0.1)
    with col_s2:
        turbidity_val = st.slider("Turbidity (NTU)", 0.0, 50.0, 4.0, 0.5)
    with col_s3:
        tds_val = st.slider("TDS (ppm)", 0.0, 1000.0, 250.0, 10.0)

st.markdown("---")

# Results Display Section
st.subheader("📊 Evaluation Metrics & Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🧪 pH Level", value=f"{ph_val:.2f}")
with col2:
    st.metric(label="🌫️ Turbidity", value=f"{turbidity_val:.2f} NTU")
with col3:
    st.metric(label="🧂 Total Dissolved Solids", value=f"{tds_val:.2f} ppm")

# Compute Fuzzy Result
score = evaluate_fuzzy_water(ph_val, turbidity_val, tds_val)

st.markdown("### 🔬 Fuzzy Inference Engine Score")
st.metric(label="Water Quality Index (WQI)", value=f"{score:.2f} / 100")

# Professional Status Cards
if score > 70:
    st.success("🟢 **Status: SAFE / POTABLE WATER** — The water quality parameters are well within safe consumption limits.")
elif score >= 40:
    st.warning("🟡 **Status: MODERATE / TREATMENT RECOMMENDED** — Water needs filtration, boiling, or purification before use.")
else:
    st.error("🔴 **Status: UNSAFE / HAZARDOUS** — High contamination risk detected. Do not consume under any circumstances.")