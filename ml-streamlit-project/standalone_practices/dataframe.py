import streamlit as st
import pandas as pd

st.title("Belajar DataFrame - AmesHousing.csv")
df = pd.read_csv("data/AmesHousing.csv")
st.dataframe(df[["Order", "PID", "Neighborhood", "Gr Liv Area", "SalePrice"]].head(10))
