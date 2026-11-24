import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
chart_data = df.groupby("Overall Qual")[["Gr Liv Area", "Total Bsmt SF", "Garage Area"]].mean().round(1).head(5)
col1, col2 = st.columns(2)
with col1:
    st.table(chart_data)
with col2:
    st.area_chart(chart_data)
