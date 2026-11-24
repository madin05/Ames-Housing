import streamlit as st

with st.expander("Klik untuk melihat Expander"):
    st.write("ini adalah contoh expander")
    st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
klik = st.expander("Klik untuk melihat Expander (Objek)")
klik.write("ini adalah contoh expander")
klik.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
