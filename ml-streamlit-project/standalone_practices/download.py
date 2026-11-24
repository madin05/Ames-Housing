import streamlit as st
import pandas as pd

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

df = pd.read_csv("data/AmesHousing.csv")[["Order", "PID", "Neighborhood", "Gr Liv Area", "SalePrice"]].head(100)
csv = convert_df(df)
st.download_button(
    label="Download data",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
)
