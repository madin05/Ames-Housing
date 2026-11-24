import streamlit as st

picture = st.camera_input("Pilih Gambar")
if picture:
    st.image(picture)
