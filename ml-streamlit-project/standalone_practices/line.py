import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
chart_data = df[df["Year Built"] >= 2000].groupby("Year Built")[["Gr Liv Area", "Total Bsmt SF", "1st Flr SF"]].mean().round(1)
st.table(chart_data)
st.line_chart(chart_data)
