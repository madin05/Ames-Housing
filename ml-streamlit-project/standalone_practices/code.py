import streamlit as st

code = """def hello():
    print("Hello, Streamlit!")"""
st.code(code, language="python")
st.code("x=2023")
