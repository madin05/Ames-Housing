import streamlit as st

with st.container():
    st.write("ini didalam container")
st.write("ini diluar container")
klik = st.container()
klik.write("ini adalah contoh container")
klik.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
