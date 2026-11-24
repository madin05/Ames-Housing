import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
chart_data = df.groupby("Bldg Type")[["1st Flr SF", "2nd Flr SF", "Total Bsmt SF"]].mean().round(1)
st.table(chart_data)
st.bar_chart(chart_data)
