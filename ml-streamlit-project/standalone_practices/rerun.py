import streamlit as st

st.title("Aplikasi Dasar Streamlit ")
nama = st.text_input("Nama")
umur = st.number_input("Umur", min_value=0, max_value=100, step=1)
rerun_button = st.button("Rerun")
if rerun_button:
    st.rerun()
st.write("Nama:", nama)
st.write("Umur:", umur)
