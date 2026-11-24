import streamlit as st

st.title("Aplikasi Dasar Streamlit Formulir")
with st.form("Formulir Data Mahasiswa"):
    col1, col2 = st.columns(2)
    with col1:
        nama = st.text_input("Nama")
        umur = st.number_input("Umur", min_value=0, max_value=100, step=1)
        alamat = st.text_area("Alamat")
        jurusan = st.selectbox("Jurusan", ["Informatika", "Sistem Informasi", "Teknik Elektro"])
    with col2:
        hobi = st.multiselect("Hobi", ["Olahraga", "Membaca", "Menulis"])
        uts = st.number_input("uts", min_value=0, max_value=100, step=1)
        uas = st.number_input("uas", min_value=0, max_value=100, step=1)
        tugas = st.number_input("tugas", min_value=0, max_value=100, step=1)
        na = (uts + uas + tugas) / 3
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    st.write("Data Mahasiswa:")
    st.write("Nama:", nama)
    st.write("Umur:", umur)
    st.write("Alamat:", alamat)
    st.write("Jurusan:", jurusan)
    st.write("Hobi:", hobi)
    st.write("Nilai Akhir", na)
