import streamlit as st

from fuzzy_logic import assess_water_quality
from llm_chain import get_ai_explanation

st.set_page_config(
    page_title="Water Quality Assessment System",
    page_icon="💧",
    layout="centered"
)

st.title("💧 Water Quality Assessment System")
st.markdown("""
### AI + Fuzzy Logic Based Water Quality Assessment

This system evaluates water quality using **Fuzzy Logic**
and provides a simple explanation using **AI with LangChain**.
""")

st.divider()

st.header("🔬 Enter Water Parameters")

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

with col2:
    tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=1000.0, value=300.0, step=10.0)

if st.button("🔍 Assess Water Quality", use_container_width=True):
    try:
        score, category = assess_water_quality(ph, turbidity, tds)
        st.session_state["score"] = score
        st.session_state["category"] = category
        st.session_state["ph"] = ph
        st.session_state["turbidity"] = turbidity
        st.session_state["tds"] = tds
    except Exception as error:
        st.error(f"An error occurred: {error}")

if "score" in st.session_state:
    score = st.session_state["score"]
    category = st.session_state["category"]

    st.divider()
    st.header("📊 Water Quality Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Quality Score", f"{score}/100")
    with col2:
        st.metric("Quality Category", category)

    st.progress(min(score / 100, 1.0))

    if category == "Excellent":
        st.success("🟢 The fuzzy system classified the water quality as Excellent.")
    elif category == "Good":
        st.success("🟢 The fuzzy system classified the water quality as Good.")
    elif category == "Moderate":
        st.warning("🟡 The fuzzy system classified the water quality as Moderate.")
    else:
        st.error("🔴 The fuzzy system classified the water quality as Poor.")

    st.subheader("Entered Parameters")
    st.table({
        "Parameter": ["pH", "Turbidity", "TDS"],
        "Value": [str(ph), f"{turbidity} NTU", f"{tds} mg/L"]
    })

st.divider()
st.header("🤖 AI Water Quality Assistant")
st.write("Ask a question about the water-quality assessment.")

question = st.text_input(
    "Ask your question",
    placeholder="Example: What does my water quality score mean?"
)

if st.button("💬 Ask AI", use_container_width=True):
    if "score" not in st.session_state:
        st.warning("⚠️ Please assess the water quality first.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤖 AI is analyzing the result..."):
            try:
                answer = get_ai_explanation(
                    st.session_state["ph"],
                    st.session_state["turbidity"],
                    st.session_state["tds"],
                    st.session_state["score"],
                    st.session_state["category"],
                    question
                )
                st.subheader("💡 AI Explanation")
                st.write(answer)
            except Exception as error:
                st.error(f"AI service error: {error}")

st.divider()

with st.expander("ℹ️ About This Project"):
    st.write("""
    **Project:** AI-Based Water Quality Assessment System

    **Technologies Used:** Python, Streamlit, Fuzzy Logic,
    scikit-fuzzy, LangChain and LLM.

    **Fuzzy Logic:** The system uses membership functions,
    fuzzy rules and defuzzification to assess water quality.

    **AI Component:** LangChain processes the user's
    natural-language question and explains the fuzzy result.

    **Note:** This is an educational project and is not a
    certified laboratory test or official drinking-water certification.
    """)

st.caption("AI + Fuzzy Logic Mini Project | Water Quality Assessment System")
