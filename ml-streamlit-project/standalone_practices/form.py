import streamlit as st

with st.form("Formulir Pengguna"):
    nama = st.text_input("Nama")
    umur = st.number_input("Umur", min_value=0, max_value=100, step=1)
    alamat = st.text_area("Alamat")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    st.write("Nama:", nama)
    st.write("Umur:", umur)
    st.write("Alamat:", alamat)
