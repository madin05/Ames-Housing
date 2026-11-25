# 🏠 Ames Housing Machine Learning & Analytics Dashboard

Dokumentasi komprehensif ini memuat perbandingan evolusi dari pengembangan aplikasi web interaktif berbasis **Streamlit** dan **Machine Learning** menggunakan dataset **Ames Housing**. Proyek ini mendokumentasikan progres dari implementasi dasar (_Basic UI_) hingga integrasi _pipeline_ pemodelan prediktif lanjutan dan _multi-module dashboard_.

---

## 📂 Perbandingan Versi & Evolusi Aplikasi

Terdapat 3 fase perkembangan skrip pada proyek praktikum ini:

| Versi Skrip                                | Deskripsi & Fokus Utama                                                                                        | Fitur & Modul Utama                                                                                                                                                                                                         |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. `app.py`** _(Versi Dasar)_            | Fokus pada implementasi awal antarmuka Streamlit (Bab 1 & 2).                                                  | • Layout dasar (`st.sidebar`, `st.columns`, `st.tabs`)<br>• Filter interaktif (`st.selectbox`, `st.slider`)<br>• Visualisasi dasar (_Bar_ & _Scatter chart_) tanpa _Machine Learning_.                                      |
| **2. `app1.py`** _(Versi Terintegrasi)_    | Menggabungkan logika _preprocessing_ data dan pemodelan regresi dalam satu halaman.                            | • Penanganan outlier & _Data Leakage_ dengan `Pipeline`<br>• Reduksi dimensi ortogonal (**PCA**)<br>• Komparasi 4 paradigma _Supervised Learning_ (Linear, KNN, Decision Tree, Random Forest).                              |
| **3. `dashboard.py`** _(Versi Multi-Modul)_ | Pengembangan tingkat lanjut dengan struktur navigasi berbasis menu berhalaman (_Multi-page/Multi-tab layout_). | • 5 modul terpisah (Overview, Diagnostic EDA 10 Masalah, PCA, Komparasi ML, Simulator Real-Time)<br>• Integrasi **Plotly Express** untuk visualisasi interaktif<br>• _Form_ interaktif untuk prediksi harga rumah seketika. |

---

## 🛠️ Perbedaan Utama Antar Versi

### 1. Dari `app.py` ➔ `app1.py` (Integrasi Pipeline & ML)

- **Penambahan Logika Machine Learning:** `app.py` murni hanya menampilkan statistik deskriptif dan grafik eksploratif. Sementara `app1.py` menambahkan _engine_ pelatihan model _Machine Learning_ (`Linear Regression`, `K-Neighbors`, `Decision Tree`, dan `Random Forest`).
- **Validasi & Pencegahan _Data Leakage_:** Menggunakan `sklearn.pipeline.Pipeline` dan `ColumnTransformer` untuk memastikan penskalaan data dan imputasi nilai hilang (_missing values_) dihitung secara terisolasi hanya pada data latih (_train set_).
- **Evaluasi Perhitungan Matrik:** Menambahkan metrik evaluasi regresi standar industri ($R^2$ Score, RMSE, dan MAE).

### 2. Dari `app1.py` ➔ `dashboard.py` (Struktur Dashboard Profesional)

- **Arsitektur Navigasi Berbasis Modul:** Mengubah struktur aplikasi menjadi sistem halaman (_multi-view dashboard_) menggunakan `st.sidebar.radio()`.
- **Visualisasi Lanjutan (Plotly):** Menggunakan pustaka `plotly.express` untuk grafik yang lebih interaktif, responsif, dan estetik dibandingkan grafik bawaan Streamlit biasa.
- **Fitur Simulator Harga Rumah Real-Time:** Menambahkan menu simulator interaktif lengkap dengan komponen `st.form()` dan `st.form_submit_button()` di mana pengguna dapat memasukkan spesifikasi properti secara manual untuk mendapatkan estimasi valuasi harga secara instan.

---

## ⚙️ Panduan Menjalankan Aplikasi

### 1. Prasyarat Pustaka (Dependencies)

Pastikan Anda telah menginstal seluruh pustaka yang dibutuhkan di dalam _virtual environment_ Anda:

```bash
pip install streamlit pandas numpy scikit-learn scipy matplotlib plotly
```

### 2. Cara Menjalankan Skrip

Arahkan terminal ke folder proyek Anda, lalu jalankan perintah Streamlit sesuai file yang ingin dieksekusi:

- **Menjalankan Versi Dasar (`app.py`):**
  ```bash
  streamlit run app.py
  ```
- **Menjalankan Versi Terintegrasi (`app1.py`):**
  ```bash
  streamlit run app1.py
  ```
- **Menjalankan Versi Multi-Modul (`dashboard.py`):**
  ```bash
  streamlit run dashboard.py
  ```

---


