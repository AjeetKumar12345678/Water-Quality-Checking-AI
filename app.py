import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Page Config
st.set_page_config(page_title="AI Water Quality Assessment System", layout="centered")

st.title("💧 AI-Based Water Quality Assessment System")
st.write("Enter your water condition description in plain text or adjust parameters below to evaluate water safety using **LangChain & Fuzzy Logic**.")

# API Key input for LangChain LLM
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

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

    # Membership Functions for pH (Acidic, Safe/Optimal, Alkaline)
    pH['acidic'] = fuzz.trapmf(pH.universe, [0, 0, 6.5, 6.8])
    pH['safe'] = fuzz.trimf(pH.universe, [6.5, 7.2, 8.5])
    pH['alkaline'] = fuzz.trapmf(pH.universe, [8.0, 8.5, 14, 14])

    # Membership Functions for Turbidity (Clear, Moderate, Murky)
    turbidity['clear'] = fuzz.trapmf(turbidity.universe, [0, 0, 4, 6])
    turbidity['moderate'] = fuzz.trimf(turbidity.universe, [5, 10, 20])
    turbidity['murky'] = fuzz.trapmf(turbidity.universe, [15, 25, 50, 50])

    # Membership Functions for TDS (Low, Moderate, High)
    tds['low'] = fuzz.trapmf(tds.universe, [0, 0, 150, 300])
    tds['moderate'] = fuzz.trimf(tds.universe, [250, 500, 750])
    tds['high'] = fuzz.trapmf(tds.universe, [700, 850, 1000, 1000])

    # Membership Functions for Quality Score (Poor, Moderate, Good)
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

    # Pass inputs
    water_sim.input['pH'] = float(np.clip(pH_val, 0, 14))
    water_sim.input['turbidity'] = float(np.clip(turbidity_val, 0, 50))
    water_sim.input['tds'] = float(np.clip(tds_val, 0, 1000))

    try:
        water_sim.compute()
        return water_sim.output['quality']
    except:
        return 50.0 # Default fallback

# Input Section
user_query = st.text_area("Describe water sample condition:", "The water looks a bit cloudy, tastes slightly metallic, and the pH is around 6.8 with TDS near 350 ppm.")

if st.button("Analyze Water Quality"):
    ph_val, turbidity_val, tds_val = 7.0, 5.0, 300.0 # defaults
    
    if api_key and user_query:
        try:
            with st.spinner("LangChain AI is extracting parameters from text..."):
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
                structured_llm = llm.with_structured_output(WaterParameters)
                extracted = structured_llm.invoke(f"Extract water test parameters from this text: {user_query}")
                ph_val = extracted.pH
                turbidity_val = extracted.turbidity
                tds_val = extracted.tds
                st.success("Parameters successfully extracted via LangChain!")
        except Exception as e:
            st.warning(f"LLM Extraction failed ({e}). Using default/fallback estimates.")

    # Show Extracted or Evaluated Parameters
    st.subheader("Extracted / Input Parameters")
    col1, col2, col3 = st.columns(3)
    col1.metric("pH Level", f"{ph_val:.2f}")
    col2.metric("Turbidity (NTU)", f"{turbidity_val:.2f}")
    col3.metric("TDS (ppm)", f"{tds_val:.2f}")

    # Compute Fuzzy Result
    score = evaluate_fuzzy_water(ph_val, turbidity_val, tds_val)

    st.subheader("Fuzzy Inference Engine Result")
    st.metric("Water Quality Index (Score out of 100)", f"{score:.2f} / 100")

    if score > 70:
        st.success("Status: **SAFE / POTABLE WATER** — The water quality parameters are within safe limits.")
    elif score >= 40:
        st.warning("Status: **MODERATE / TREATMENT RECOMMENDED** — Water needs filtration or boiling before use.")
    else:
        st.error("Status: **UNSAFE / HAZARDOUS** — High contamination risk detected. Do not consume.")