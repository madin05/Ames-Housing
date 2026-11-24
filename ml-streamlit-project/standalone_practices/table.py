import streamlit as st
import pandas as pd

st.title("Belajar Membuat Tabel - AmesHousing.csv")
df = pd.read_csv("data/AmesHousing.csv")
st.table(df[["Neighborhood", "Bldg Type", "Overall Qual", "Gr Liv Area", "SalePrice"]].head(10))
