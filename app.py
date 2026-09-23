import streamlit as st

from fuzzy_logic import assess_water_quality
from llm_chain import get_ai_explanation


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Water Quality Assessment System",
    page_icon="💧",
    layout="centered"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("💧 Water Quality Assessment System")

st.markdown(
    """
    ### AI + Fuzzy Logic Based Water Quality Assessment

    This system evaluates water quality using **Fuzzy Logic**
    and provides a simple explanation using **AI with LangChain**.
    """
)

st.divider()


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.header("🔬 Enter Water Parameters")

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1
    )

    turbidity = st.number_input(
        "Turbidity (NTU)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1
    )

with col2:
    tds = st.number_input(
        "TDS (mg/L)",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=10.0
    )


# ---------------------------------------------------------
# Assessment button
# ---------------------------------------------------------

if st.button(
    "🔍 Assess Water Quality",
    use_container_width=True
):
    try:
        score, category = assess_water_quality(
            ph,
            turbidity,
            tds
        )

        # Store results
        st.session_state["score"] = score
        st.session_state["category"] = category
        st.session_state["ph"] = ph
        st.session_state["turbidity"] = turbidity
        st.session_state["tds"] = tds

        st.success("Water quality assessment completed.")

    except Exception as error:
        st.error(
            f"Unable to calculate water quality: {error}"
        )


# ---------------------------------------------------------
# Display assessment result
# ---------------------------------------------------------

if "score" in st.session_state:

    score = st.session_state["score"]
    category = st.session_state["category"]

    ph_result = st.session_state["ph"]
    turbidity_result = st.session_state["turbidity"]
    tds_result = st.session_state["tds"]

    st.divider()

    st.header("📊 Water Quality Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Quality Score",
            f"{score}/100"
        )

    with col2:
        st.metric(
            "Quality Category",
            category
        )

    # Progress bar
    st.progress(
        min(max(score / 100, 0.0), 1.0)
    )


    # Category message
    if category == "Excellent":

        st.success(
            "🟢 The fuzzy system classified the "
            "water quality as Excellent."
        )

    elif category == "Good":

        st.success(
            "🟢 The fuzzy system classified the "
            "water quality as Good."
        )

    elif category == "Moderate":

        st.warning(
            "🟡 The fuzzy system classified the "
            "water quality as Moderate."
        )

    else:

        st.error(
            "🔴 The fuzzy system classified the "
            "water quality as Poor."
        )


    # -----------------------------------------------------
    # Parameters table
    # -----------------------------------------------------

    st.subheader("📋 Entered Parameters")

    st.table(
        {
            "Parameter": [
                "pH",
                "Turbidity",
                "TDS"
            ],
            "Value": [
                f"{ph_result:.1f}",
                f"{turbidity_result:.1f} NTU",
                f"{tds_result:.0f} mg/L"
            ]
        }
    )


# ---------------------------------------------------------
# AI section
# ---------------------------------------------------------

st.divider()

st.header("🤖 AI Water Quality Assistant")

st.write(
    "Ask a question about the water-quality assessment."
)

question = st.text_input(
    "Ask your question",
    placeholder=(
        "Example: What does my water quality score mean?"
    )
)


if st.button(
    "💬 Ask AI",
    use_container_width=True
):

    if "score" not in st.session_state:

        st.warning(
            "⚠️ Please assess the water quality first."
        )

    elif not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

    else:

        with st.spinner(
            "🤖 AI is analyzing the result..."
        ):

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

                st.error(
                    f"AI service error: {error}"
                )


# ---------------------------------------------------------
# About project
# ---------------------------------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        **Project:** AI-Based Water Quality Assessment System

        **Technologies Used:**
        Python, Streamlit, Fuzzy Logic,
        scikit-fuzzy, LangChain and OpenAI.

        **Fuzzy Logic:**
        The system uses membership functions,
        fuzzy rules and defuzzification to calculate
        a water-quality score.

        **AI Component:**
        LangChain sends the assessment information
        and the user's question to an OpenAI language
        model and generates a simple explanation.

        **Important Note:**
        This is an educational project. The fuzzy score
        is not a certified laboratory test and should not
        be treated as official drinking-water certification.
        """
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.caption(
    "AI + Fuzzy Logic Mini Project | "
    "Water Quality Assessment System"
)
