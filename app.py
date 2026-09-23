import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Page Configuration
st.set_page_config(
    page_title="Water Quality Assessment AI",
    page_icon="💧",
    layout="centered"
)

# Custom CSS for modern styling and attractive look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 45px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #004080;
        color: #ffffff;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💧 AI-Powered Water Quality Assessment")
st.markdown("Evaluate water safety instantly using **Google Gemini AI** and **Fuzzy Logic Control Systems**.")

# Retrieve Gemini API Key securely from Streamlit Secrets or Sidebar fallback
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_input = st.text_input("Gemini API Key", value=api_key, type="password", help="Enter your Google AI Studio Gemini API Key")
    if user_api_input:
        api_key = user_api_input
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("You can type descriptions like: *'Water is slightly murky, tastes metallic, pH is around 6.5'*, and AI will automatically extract the metrics!")

# Pydantic schema for LangChain structured extraction
class WaterParameters(BaseModel):
    pH: float = Field(description="pH level of the water, typically between 0 and 14")
    turbidity: float = Field(description="Turbidity level in NTU (Nephelometric Turbidity Units)")
    tds: float = Field(description="Total Dissolved Solids in ppm or mg/L")

# Fuzzy Logic Calculation Engine
def evaluate_fuzzy_water(pH_val, turbidity_val, tds_val):
    # Antecedents (Inputs)
    pH = ctrl.Antecedent(np.arange(0, 14.1, 0.1), 'pH')
    turbidity = ctrl.Antecedent(np.arange(0, 51, 1), 'turbidity')
    tds = ctrl.Antecedent(np.arange(0, 1001, 1), 'tds')

    # Consequent (Output)
    quality = ctrl.Consequent(np.arange(0, 101, 1), 'quality')

    # Membership Functions for pH
    pH['acidic'] = fuzz.trapmf(pH.universe, [0, 0, 6.5, 6.8])
    pH['safe'] = fuzz.trimf(pH.universe, [6.5, 7.2, 8.5])
    pH['alkaline'] = fuzz.trapmf(pH.universe, [8.0, 8.5, 14, 14])

    # Membership Functions for Turbidity
    turbidity['clear'] = fuzz.trapmf(turbidity.universe, [0, 0, 4, 6])
    turbidity['moderate'] = fuzz.trimf(turbidity.universe, [5, 10, 20])
    turbidity['murky'] = fuzz.trapmf(turbidity.universe, [15, 25, 50, 50])

    # Membership Functions for TDS
    tds['low'] = fuzz.trapmf(tds.universe, [0, 0, 150, 300])
    tds['moderate'] = fuzz.trimf(tds.universe, [250, 500, 750])
    tds['high'] = fuzz.trapmf(tds.universe, [700, 850, 1000, 1000])

    # Membership Functions for Quality Score
    quality['poor'] = fuzz.trimf(quality.universe, [0, 0, 40])
    quality['moderate'] = fuzz.trimf(quality.universe, [30, 50, 70])
    quality['good'] = fuzz.trimf(quality.universe, [60, 100, 100])

    # Fuzzy Rules
    rule1 = ctrl.Rule(pH['safe'] & turbidity['clear'] & tds['low'], quality['good'])
    rule2 = ctrl.Rule(pH['acidic'] | turbidity['murky'] | tds['high'], quality['poor'])
    rule3 = ctrl.Rule(turbidity['moderate'] | tds['moderate'], quality['moderate'])
    rule4 = ctrl.Rule(pH['alkaline'], quality['moderate'])

    # Control System Simulation
    quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    water_sim = ctrl.ControlSystemSimulation(quality_ctrl)

    # Pass inputs with clipping
    water_sim.input['pH'] = float(np.clip(pH_val, 0, 14))
    water_sim.input['turbidity'] = float(np.clip(turbidity_val, 0, 50))
    water_sim.input['tds'] = float(np.clip(tds_val, 0, 1000))

    try:
        water_sim.compute()
        return water_sim.output['quality']
    except Exception:
        return 50.0  # Default fallback score

# Input Section
user_query = st.text_area(
    "📝 Describe water sample condition:",
    "The water looks a bit cloudy, tastes slightly metallic, and the pH is around 6.8 with TDS near 350 ppm.",
    height=100
)

if st.button("🚀 Analyze Water Quality"):
    ph_val, turbidity_val, tds_val = 7.0, 5.0, 300.0  # Default fallback values
    
    if api_key and user_query:
        try:
            with st.spinner("✨ Gemini AI is analyzing and extracting water parameters..."):
                # Using Google Gemini via LangChain ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
                structured_llm = llm.with_structured_output(WaterParameters)
                extracted = structured_llm.invoke(f"Extract water test parameters accurately from this text: {user_query}")
                
                ph_val = extracted.pH
                turbidity_val = extracted.turbidity
                tds_val = extracted.tds
                st.success("Parameters successfully extracted via Gemini AI!")
        except Exception as e:
            st.warning(f"AI Extraction warning ({e}). Using estimated default parameter values.")
    elif not api_key:
        st.warning("⚠️ Gemini API key not detected. Using default fallback parameter values. Add your key in the sidebar or Streamlit secrets.")

    # Show Extracted or Evaluated Parameters in Clean Metrics
    st.markdown("### 📊 Extracted / Evaluated Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🧪 pH Level", value=f"{ph_val:.2f}")
    with col2:
        st.metric(label="🌫️ Turbidity", value=f"{turbidity_val:.2f} NTU")
    with col3:
        st.metric(label="🧂 TDS", value=f"{tds_val:.2f} ppm")

    # Compute Fuzzy Result
    score = evaluate_fuzzy_water(ph_val, turbidity_val, tds_val)

    st.markdown("### 🔬 Fuzzy Inference Engine Result")
    st.metric(label="Water Quality Index (Score out of 100)", value=f"{score:.2f} / 100")

    # Status Cards Presentation
    if score > 70:
        st.success("🟢 **Status: SAFE / POTABLE WATER** — The water quality parameters are well within safe consumption limits.")
    elif score >= 40:
        st.warning("🟡 **Status: MODERATE / TREATMENT RECOMMENDED** — Water needs filtration, boiling, or purification before use.")
    else:
        st.error("🔴 **Status: UNSAFE / HAZARDOUS** — High contamination risk detected. Do not consume under any circumstances.")