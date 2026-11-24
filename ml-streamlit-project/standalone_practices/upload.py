import streamlit as st

upload = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
st.write("file yang diupload", upload)
