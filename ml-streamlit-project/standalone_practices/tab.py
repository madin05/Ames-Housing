import streamlit as st
import pandas as pd

df = pd.read_csv("data/AmesHousing.csv")
data = df.groupby("Yr Sold")["SalePrice"].mean()
tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
tab1.subheader("contoh tab dengan grafik")
tab1.line_chart(data)
tab2.subheader("contoh tab dengan data ")
tab2.write(data)
