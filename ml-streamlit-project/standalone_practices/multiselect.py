import streamlit as st

options = st.multiselect("Hobby Kamu ?", ["Swimming", "Basket", "Soccer"])
st.write("Kamu Memilih:", options)
