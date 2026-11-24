import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/AmesHousing.csv")
fig, ax = plt.subplots()
ax.hist(df["SalePrice"].dropna(), bins=25, color="#2563eb", edgecolor="white")
ax.set_title("Distribusi SalePrice Ames Housing")
st.pyplot(fig)
