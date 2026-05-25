import streamlit as st
from agents import (
    subject_analyzer_agent,
    answer_agent
)

st.title("EngineeringGuru - Mumbai University")
st.write("Welcome to EngineeringGuru, your ultimate study companion for Mumbai University! Whether you're a student looking to ace your exams or a lifelong learner eager to explore new subjects, our AI-powered platform is here to help you succeed")

subject = st.sidebar.text_input("Enter Subject (e.g., DBMMS, CN):")
if subject:
    with st.spinner("Analyzing subject..."):
        analyzed = subject_analysis = subject_analyzer_agent(subject)
    st.subheader("🔍Subject Analysis")
    st.write(subject_analysis)

    with st.spinner("Generating questions and answers..."):
        qna = answer_agent(analyzed)
    st.subheader("📖 Important Questions and Answers")
    st.write(qna)

else:
    st.info("Please enter a subject code in the sidebar to get started.")