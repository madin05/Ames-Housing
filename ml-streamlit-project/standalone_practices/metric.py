import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
st.caption("dibawah ini kombinasi coding st.metric() dengan st.columns()")
st.divider()
kolom1, kolom2, kolom3 = st.columns(3)
kolom1.metric("Mean SalePrice", f"${df['SalePrice'].mean():,.0f}", "+$20,796 vs Median")
kolom2.metric("Mean Gr Liv Area", f"{df['Gr Liv Area'].mean():,.1f} sqft", "+57.7 sqft")
kolom3.metric("Mean Overall Qual", f"{df['Overall Qual'].mean():.2f}", "+0.09")
