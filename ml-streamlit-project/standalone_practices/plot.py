import plotly.express as px
import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv").dropna(subset=["Gr Liv Area", "SalePrice", "Lot Area", "Bldg Type", "Neighborhood"]).head(200)
fig = px.scatter(
    df,
    x="Gr Liv Area",
    y="SalePrice",
    size="Lot Area",
    color="Bldg Type",
    hover_name="Neighborhood",
    log_x=True,
    size_max=60,
)
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)
