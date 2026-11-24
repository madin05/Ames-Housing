import streamlit as st

tinggi = st.slider("Berapa Tinggi Kamu?", 0, 300, 170)
st.write("Tinggi Saya ", tinggi, "cm")
nilai = st.slider("Pilih Interval 0-1", min_value=0.0, max_value=1.0, step=0.1)
st.write("Nilai Interval yang dipilih adalah :", nilai)
