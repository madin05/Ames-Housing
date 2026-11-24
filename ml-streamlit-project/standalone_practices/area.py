import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
chart_data = df.groupby("Overall Qual")[["Gr Liv Area", "Total Bsmt SF", "Garage Area"]].mean().round(1)
st.table(chart_data)
st.area_chart(chart_data)
