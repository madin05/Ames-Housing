import streamlit as st

na = st.text_input("Nama :")
st.write("nama yang diinputkan ", na)
nomor = st.number_input("pilihlah angka : ", min_value=0, max_value=10)
st.write("angka yang dipilih ", nomor)
txt = st.text_area("Masukkan Alamat")
st.write("Inputan Textarea:", txt)
