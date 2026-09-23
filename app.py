import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Page Configuration
st.set_page_config(
    page_title="Water Quality Assessment System",
    page_icon="💧",
    layout="wide"
)

# Professional Creative Water Theme CSS (Glassmorphism & Clean UI)
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
st.title("💧 Advanced Water Quality Assessment System")
st.markdown("##### Professional Water Safety Analytics powered by **Fuzzy Logic Control Systems**")
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/water.png", width=70)
    st.header("🌊 Dashboard Info")
    st.info("This system evaluates water potability and safety metrics instantly using precise fuzzy mathematics without external cloud dependencies.")
    st.markdown("---")
    st.markdown("### 📊 Status Legend")
    st.success("🟢 Safe (>70)")
    st.warning("🟡 Moderate (40-70)")
    st.error("🔴 Unsafe (<40)")

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

# New Feature Tabs: Preset Water Samples vs Manual Custom Sliders
tab1, tab2 = st.tabs(["🧪 Quick Water Type Presets", "🎛️ Custom Parameter Sliders"])

ph_val, turbidity_val, tds_val = 7.2, 3.0, 200.0

with tab1:
    st.subheader("Select Standard Water Sample Type")
    st.markdown("Choose from standard pre-configured water profiles to instantly analyze typical category safety:")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    preset_choice = None
    with col_p1:
        if st.button("🚰 Municipal Tap Water"):
            preset_choice = (7.4, 2.0, 180.0)
    with col_p2:
        if st.button("🏞️ Borewell / Groundwater"):
            preset_choice = (6.8, 12.0, 650.0)
    with col_p3:
        if st.button("🍾 Bottled Mineral Water"):
            preset_choice = (7.0, 0.5, 90.0)
    with col_p4:
        if st.button("🌊 Contaminated / River"):
            preset_choice = (5.5, 35.0, 850.0)

    # Session state to hold preset or slider values
    if "water_vals" not in st.session_state:
        st.session_state.water_vals = (7.2, 3.0, 200.0)

    if preset_choice:
        st.session_state.water_vals = preset_choice
        st.success("Preset sample loaded successfully!")

    ph_val, turbidity_val, tds_val = st.session_state.water_vals

with tab2:
    st.subheader("Fine-Tune Parameters Manually")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        ph_val = st.slider("pH Level (0 - 14)", 0.0, 14.0, float(ph_val), 0.1)
    with col_s2:
        turbidity_val = st.slider("Turbidity (NTU)", 0.0, 50.0, float(turbidity_val), 0.5)
    with col_s3:
        tds_val = st.slider("TDS (ppm)", 0.0, 1000.0, float(tds_val), 10.0)
    
    st.session_state.water_vals = (ph_val, turbidity_val, tds_val)

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