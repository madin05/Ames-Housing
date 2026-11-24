# Implementasi Praktik Modul Machine Learning Bab 1–2 dengan Streamlit

Proyek ini merupakan implementasi teknis seluruh tahapan praktik pada **Bab 1 (Pengenalan Framework Web Streamlit)** dan **Bab 2 (Membuat Aplikasi Dasar Streamlit)** dari buku *Implementasi Algoritma Machine Learning Berbasis Web dengan Framework Streamlit* (Abu Tholib, M.Kom, 2023) menggunakan dataset kelompok **`AmesHousing.csv`**.

## Struktur Proyek

```text
ml-streamlit-project/
├── app.py                        # Aplikasi utama Streamlit terintegrasi (Bab 1 & Bab 2 + Audit Dataset)
├── IMPLEMENTATION_CHECKLIST.md   # Pemetaan 47 instruksi Bab 1–2 beserta halaman buku/PDF & status
├── verify_implementation.py      # Skrip verifikasi otomatis, pembuatan aset, log audit, & standalone files
├── data/
│   └── AmesHousing.csv           # Salinan kerja dataset kelompok (2.930 baris × 82 kolom)
├── assets/
│   ├── bimbingan.jpg             # Aset gambar untuk praktik Bab 2.6.a (st.image)
│   └── sample_audio.wav          # Aset suara untuk praktik Bab 2.6.b (st.audio)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Pemuatan dataset, validasi CSV, audit struktur & pemetaan 82 kolom
│   ├── preprocessing.py          # Pemeriksaan kualitas data, outlier, identifier, & pencegahan data leakage
│   └── analysis.py               # Penyiapan data tabel, metrik, grafik Bab 2.3–2.4 & pembuatan aset
├── standalone_practices/         # Seluruh file latihan mandiri sesuai penamaan buku Bab 1 & Bab 2
├── outputs/
│   ├── screenshots/              # Bukti tangkapan layar aktual aplikasi Streamlit yang dijalankan
│   ├── figures/                  # Grafik hasil analisis yang diekspor
│   └── logs/                     # Log hasil audit dataset, pengujian fungsi, & environment
├── requirements.txt              # Daftar dependensi proyek
└── README.md                     # Dokumentasi penggunaan proyek
```

## Cara Menjalankan Aplikasi

1. Aktifkan virtual environment dari root workspace:
   ```powershell
   & "c:\Arif\machine learning\.venv\Scripts\Activate.ps1"
   ```
2. Masuk ke direktori proyek dan jalankan aplikasi utama Streamlit:
   ```powershell
   cd "c:\Arif\machine learning\ml-streamlit-project"
   streamlit run app.py
   ```
3. Untuk menjalankan skrip verifikasi otomatis dan menghasilkan log audit ke `outputs/logs/`:
   ```powershell
   & "c:\Arif\machine learning\.venv\Scripts\python.exe" verify_implementation.py
   ```

