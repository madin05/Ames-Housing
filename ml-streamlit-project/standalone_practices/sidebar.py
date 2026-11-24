import streamlit as st

pilih = st.sidebar.selectbox(
    "pilihlah algoritma yang ingin digunakan ?",
    ("SVM", "KNN", "K-Means"),
)
with st.sidebar:
    pilih_radio = st.radio(
        "Pilihlah type Data",
        ("Numerik", "Kategorik"),
    )
