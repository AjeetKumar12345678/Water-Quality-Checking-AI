# 💧 Water Quality Assessment System

AI + Fuzzy Logic based mini project using Python.

## Features

- Water quality assessment using Fuzzy Logic
- Membership functions
- Fuzzy inference rules
- Defuzzification
- Streamlit web interface
- LangChain-based AI explanation
- Natural-language questions

## Parameters

- pH
- Turbidity
- TDS

## Project Structure

```text
Water-Quality-Assessment-System/
├── app.py
├── fuzzy_logic.py
├── llm_chain.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API Key

Create a `.env` file locally:

```text
OPENAI_API_KEY=your_api_key_here
```

Never upload `.env` to GitHub.

## Disclaimer

This is an educational mini project. The fuzzy score is not a certified
laboratory test or official drinking-water safety certification.
