import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
source = df[["Gr Liv Area", "SalePrice", "Bldg Type"]].dropna().head(150)
chart = {
    "mark": "point",
    "encoding": {
        "x": {"field": "Gr Liv Area", "type": "quantitative"},
        "y": {"field": "SalePrice", "type": "quantitative"},
        "color": {"field": "Bldg Type", "type": "nominal"},
        "shape": {"field": "Bldg Type", "type": "nominal"},
    },
}
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Vega-Lite native theme"])
with tab1:
    st.vega_lite_chart(source, chart, theme="streamlit", use_container_width=True)
with tab2:
    st.vega_lite_chart(source, chart, theme=None, use_container_width=True)
