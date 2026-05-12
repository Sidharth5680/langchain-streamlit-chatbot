import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI chatbot.

    User Question:
    {question}
    """
)

chain = prompt | llm

st.title("Basic AI Chatbot Using LangChain")

user_input = st.text_input("Ask a question:")

if user_input:
    response = chain.invoke({"question": user_input})

    st.write("### Chatbot Response:")
    st.write(response.content)