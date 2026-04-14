from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    temperature=0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

def run_agent(prompt):
    return llm.invoke([HumanMessage(content=prompt)]).content


# 🔹 Understand Project (VERY IMPORTANT)
def subject_analyzer_agent(subject_text):
    prompt = f"""
    Trends from: https://muquestionpapers.com/   
    Analyze the {subject_text} provided as per the Mumbai University website with respect to new syllabus after 2020.
        Provide:
        - Syllabus breakdown
        - Important topics
        - Weightage of each topic in exams with respect to each module
        - Any patterns or trends observed in previous years' question papers
    ANALYSIS:
    {subject_text}
    """
    return run_agent(prompt)

def answer_agent(subject_text):
    prompt = f"""
    You are a Mumbai University professor.

    Your task is to generate exam-oriented content STRICTLY based on:
    - Previous year Mumbai University question papers
    - Trends from: https://muquestionpapers.com/

    ⚠️ Instructions:
    - Generate 15 MOST IMPORTANT questions both long and short
    - Questions must reflect:
        • Repeated questions
        • Long questions (10 marks, 15 questions)
        • Short questions (5 marks, 15 questions)
    - Follow MU exam pattern and trends closely

    SUBJECT:
    {subject_text}
    """
    return run_agent(prompt)