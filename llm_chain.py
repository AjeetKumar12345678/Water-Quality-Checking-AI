import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()


# =========================================================
# Create LangChain LLM Chain
# =========================================================

def create_llm_chain():

    api_key = os.getenv("OPENAI_API_KEY")


    # -----------------------------------------------------
    # API key is optional.
    # The fuzzy system can work without the AI component.
    # -----------------------------------------------------

    if not api_key:

        return None


    # -----------------------------------------------------
    # Create OpenAI model
    # -----------------------------------------------------

    llm = ChatOpenAI(
        model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini"
        ),
        temperature=0.2,
        api_key=api_key
    )


    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a Water Quality Assessment AI assistant.

The fuzzy logic system has already calculated a
water-quality score.

Water Parameters:

pH: {ph}

Turbidity: {turbidity} NTU

TDS: {tds} mg/L

Fuzzy Quality Score: {score}/100

Fuzzy Category: {category}

User Question:
{question}


Your task:

Explain the fuzzy assessment in simple and
easy-to-understand language.

Please include:

1. What the fuzzy score and category mean.
2. Which water parameters influenced the result.
3. A practical recommendation.
4. A short explanation related to the user's question.


Important:

- Do not claim that the fuzzy result is a certified
  laboratory drinking-water test.
- Do not claim that the water is medically or officially
  certified safe.
- Explain that the result is an educational fuzzy-logic
  assessment.
- Keep the answer concise and understandable.
"""
    )


    # -----------------------------------------------------
    # Build LangChain pipeline
    # -----------------------------------------------------

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =========================================================
# Get AI Explanation
# =========================================================

def get_ai_explanation(
    ph,
    turbidity,
    tds,
    score,
    category,
    question
):

    chain = create_llm_chain()


    # -----------------------------------------------------
    # No API key
    # -----------------------------------------------------

    if chain is None:

        return (
            "🤖 AI explanation is currently unavailable "
            "because OPENAI_API_KEY is not configured.\n\n"
            f"Your fuzzy water-quality score is "
            f"{score}/100 and the category is "
            f"**{category}**.\n\n"
            "You can still use the fuzzy-logic assessment "
            "without the AI component."
        )


    # -----------------------------------------------------
    # Call LangChain
    # -----------------------------------------------------

    response = chain.invoke(
        {
            "ph": ph,
            "turbidity": turbidity,
            "tds": tds,
            "score": score,
            "category": category,
            "question": question
        }
    )


    return response
