import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def create_llm_chain():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template("""
You are a Water Quality Assessment AI assistant.

The fuzzy logic system has already assessed the water.

Water Parameters:
pH: {ph}
Turbidity: {turbidity} NTU
TDS: {tds} mg/L

Fuzzy Quality Score: {score}/100
Fuzzy Category: {category}

User Question:
{question}

Explain the result in simple language.

Mention:
1. What the fuzzy result means.
2. Which parameters affected the result.
3. A practical recommendation.

Do not claim that the result is a certified laboratory drinking-water
safety test.
""")

    return prompt | llm | StrOutputParser()


def get_ai_explanation(ph, turbidity, tds, score, category, question):
    chain = create_llm_chain()

    if chain is None:
        return (
            "AI explanation is unavailable because OPENAI_API_KEY "
            "is not configured."
        )

    return chain.invoke({
        "ph": ph,
        "turbidity": turbidity,
        "tds": tds,
        "score": score,
        "category": category,
        "question": question
    })
