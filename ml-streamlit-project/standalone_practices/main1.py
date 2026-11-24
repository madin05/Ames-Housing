import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
st.write(1234)
st.write(df[["Order", "PID", "Neighborhood", "Gr Liv Area", "SalePrice"]].head(4))
