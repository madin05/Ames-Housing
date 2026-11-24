import streamlit as st

a = st.date_input("Kapan penerbangan Anda berikutnya")
st.write("waktu penerbangan", a)
b = st.time_input("Kapan waktu kelas Anda")
st.write("waktu penerbangan", b)
