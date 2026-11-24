import streamlit as st
import pandas as pd
import altair as alt

df = pd.read_csv("data/AmesHousing.csv")
chart_data = df[["Gr Liv Area", "SalePrice", "Garage Area", "Overall Qual"]].dropna().head(150)
c = alt.Chart(chart_data).mark_circle().encode(
    x="Gr Liv Area", y="SalePrice", size="Garage Area", color="Overall Qual",
    tooltip=["Gr Liv Area", "SalePrice", "Garage Area", "Overall Qual"]
)
st.altair_chart(c, use_container_width=True)
