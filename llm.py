from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the user's questions."),
        ("user", "Question: {question}")
    ]
)

st.title("Chatbot using Gemma-2b")
input_txt = st.text_input("What do you want to ask?")

model = Ollama(model="gemma:2b")
parser = StrOutputParser()

chain = prompt | model | parser

if input_txt:
    st.write(chain.invoke({"question": input_txt}))