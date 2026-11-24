import streamlit as st

st.title("Aplikasi Streamlit")
if st.button("Stop Aplikasi"):
    st.warning("Aplikasi dihentikan!")
    st.stop()
st.write("Baris ini tidak akan dieksekusi jika aplikasi dihentikan.")
