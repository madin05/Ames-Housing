import streamlit as st
import pandas as pd

st.title("Aplikasi Pertama Web Data Mining dengan Streamlit")
st.write("oleh Abu Tholib, M.Kom — Diadaptasi untuk Dataset Kelompok AmesHousing.csv")
df = pd.read_csv("data/AmesHousing.csv")
st.write(f"Dataset berhasil dimuat: {len(df)} baris dan {df.shape[1]} kolom.")
