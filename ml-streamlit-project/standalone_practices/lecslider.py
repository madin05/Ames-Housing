import streamlit as st

opsi = ["sapi", "ayam", "burung"]
pilih = st.select_slider("Hewan peliharaan", options=opsi)
st.write("pilihan kamu", pilih)
rentang = range(0, 11)
pilihan = st.select_slider("Rentang Nilai", options=rentang, value=5)
st.write("pilihan kamu", pilihan)
