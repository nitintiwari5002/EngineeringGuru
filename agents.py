from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

# ==============================
# GROQ LLM CONFIGURATION
# ==============================

llm = ChatGroq(
    temperature=0.1,  # Lower temp = more precise + less hallucination
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ==============================
# COMMON RUNNER
# ==============================

def run_agent(prompt):
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    return response.content


# =========================================================
# 🔹 SUBJECT ANALYZER AGENT
# =========================================================

def subject_analyzer_agent(subject_text):

    prompt = f"""
You are an EXPERT Mumbai University syllabus and examination analyst.

STRICT RULES:
- ONLY analyze according to Mumbai University (MU) syllabus pattern.
- Focus ONLY on revised syllabus after 2020.
- STRICTLY follow MU examination trends.
- Use concise and academic formatting.
- DO NOT generate generic textbook content.
- DO NOT provide assumptions unrelated to MU.
- Focus on repeated and high-weightage topics.
- Prioritize previous year question paper trends.

Reference:
https://muquestionpapers.com/

YOUR TASK:
Analyze the subject thoroughly and provide:

1. Subject Overview
2. Module-wise syllabus breakdown
3. Important topics per module
4. Expected weightage of each module
5. Frequently repeated questions/topics
6. Important derivations/theory/numericals
7. MU exam trends observed
8. Most important modules for scoring
9. Predicted high-probability topics for next exam

OUTPUT FORMAT STRICTLY:

SUBJECT OVERVIEW:
- ...

MODULE-WISE ANALYSIS:

MODULE 1:
- Important Topics:
- Expected Weightage:
- Repeated Questions:
- Important Theory:

MODULE 2:
...

EXAM TRENDS:
- ...

HIGH SCORING TOPICS:
- ...

PREDICTED IMPORTANT TOPICS:
- ...

SUBJECT:
{subject_text}
"""

    return run_agent(prompt)


# =========================================================
# 🔹 QUESTION PREDICTION AGENT
# =========================================================

def answer_agent(subject_text):

    prompt = f"""
You are a senior Mumbai University professor and paper setter.

STRICT INSTRUCTIONS:

- STRICTLY follow Mumbai University question paper pattern.
- ONLY generate questions relevant to revised MU syllabus after 2020.
- Use trends from previous MU papers.
- Questions MUST resemble actual MU exam style.
- Avoid generic AI-generated textbook questions.
- Prioritize:
    • Repeated questions
    • Frequently asked theory
    • Important numericals
    • Important derivations
    • Module-wise balance
    • High probability exam questions

Reference:
https://muquestionpapers.com/

GENERATE:

1. 15 IMPORTANT LONG QUESTIONS
    - 10 Marks
    - Descriptive
    - Theory/Numerical/Derivation based
    - Exam-oriented

2. 15 IMPORTANT SHORT QUESTIONS
    - 5 Marks
    - Direct conceptual questions
    - Definitions
    - Short derivations
    - Applications

STRICT OUTPUT FORMAT:

========================================
MUMBAI UNIVERSITY IMPORTANT QUESTIONS
========================================

SUBJECT: {subject_text}

----------------------------------------
SECTION A — LONG QUESTIONS (10 MARKS)
----------------------------------------

Q1.
Q2.
Q3.
...
Q15.

----------------------------------------
SECTION B — SHORT QUESTIONS (5 MARKS)
----------------------------------------

Q1.
Q2.
Q3.
...
Q15.

ADDITIONAL RULES:
- Maintain professional exam formatting.
- Keep questions concise and realistic.
- Avoid duplicate questions.
- Cover all modules proportionally.
- Prioritize high-frequency MU questions.

SUBJECT:
{subject_text}
"""

    return run_agent(prompt)


# =========================================================
# 🔹 OPTIONAL JSON OUTPUT AGENT (BEST FOR APIs)
# =========================================================

def structured_prediction_agent(subject_text):

    prompt = f"""
You are an expert Mumbai University examination analyst.

STRICTLY return ONLY valid JSON.

TASK:
Analyze Mumbai University trends and predict important exam questions.

FORMAT:

{{
    "subject_name": "",
    "important_modules": [],
    "high_weightage_topics": [],
    "repeated_topics": [],
    "predicted_long_questions": [],
    "predicted_short_questions": [],
    "exam_trends": [],
    "scoring_topics": []
}}

SUBJECT:
{subject_text}
"""

    response = run_agent(prompt)

    try:
        return json.loads(response)
    except:
        return response