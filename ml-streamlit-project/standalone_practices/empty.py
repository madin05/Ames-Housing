import streamlit as st
import time

with st.empty():
    for seconds in range(3):
        st.write(f"⏳ {seconds} detik akan selesai ")
        time.sleep(0.5)
    st.write("✔️ 3 detik sudah selesai!")
klik = st.empty()
for detik in range(3):
    klik.write(f"⏳ {detik} detik kedua akan selesai ")
    time.sleep(0.5)
klik.write("✔️ 3 detik kedua sudah selesai!")
