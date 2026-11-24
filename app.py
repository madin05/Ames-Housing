import streamlit as st
import pandas as pd

# ==========================================
# BAB 2.7: Menampilkan Layout Dasar
# ==========================================
st.set_page_config(page_title="Ames Housing ML App", layout="wide")

# ==========================================
# BAB 2.2: Elemen Text
# ==========================================
st.title(':material/home: Aplikasi Analisis Data Ames Housing')
st.markdown('**Tugas Praktikum Machine Learning - Pertemuan 14**')
st.caption('Implementasi Modul Bab 1 & 2 Framework Streamlit')
st.divider()

# ==========================================
# Load Data (Persiapan untuk Bab 2.3)
# ==========================================
@st.cache_data
def load_data():
    # Pastikan file AmesHousing.csv berada di folder yang sama
    return pd.read_csv('AmesHousing.csv')

try:
    df = load_data()
    
    # ==========================================
    # BAB 2.5 & 2.7: Menampilkan Layout Sidebar & Input Form
    # ==========================================
    with st.sidebar:
        st.header(":material/tune: Filter Data")
        
        # Contoh penggunaan Checkbox
        tampilkan_tabel = st.checkbox("Tampilkan Dataframe", value=True)
        
        # Contoh penggunaan Selectbox 
        kolom_kategori = 'Neighborhood' if 'Neighborhood' in df.columns else df.columns[1]
        pilihan_kategori = ['Semua'] + list(df[kolom_kategori].dropna().unique())
        filter_kategori = st.selectbox(f"Pilih {kolom_kategori}", pilihan_kategori)
        
        # Contoh penggunaan Slider
        if 'SalePrice' in df.columns:
            min_price = int(df['SalePrice'].min())
            max_price = int(df['SalePrice'].max())
            filter_harga = st.slider(
                "Rentang SalePrice ($)", 
                min_value=min_price, 
                max_value=max_price, 
                value=(min_price, max_price)
            )
        else:
            filter_harga = None

    # Logika Filtering Data
    df_filtered = df.copy()
    if filter_kategori != 'Semua':
        df_filtered = df_filtered[df_filtered[kolom_kategori] == filter_kategori]
    if filter_harga and 'SalePrice' in df.columns:
        df_filtered = df_filtered[(df_filtered['SalePrice'] >= filter_harga[0]) & (df_filtered['SalePrice'] <= filter_harga[1])]

    # ==========================================
    # BAB 2.3: Menampilkan Data & Metric
    # ==========================================
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Data (Baris)", value=len(df_filtered))
    
    if 'SalePrice' in df.columns:
        rata_harga = df_filtered['SalePrice'].mean()
        col2.metric(label="Rata-rata Harga", value=f"${rata_harga:,.0f}")
        max_harga = df_filtered['SalePrice'].max()
        col3.metric(label="Harga Tertinggi", value=f"${max_harga:,.0f}")

    if tampilkan_tabel:
        st.subheader(":material/table_chart: Dataset Ames Housing")
        st.dataframe(df_filtered.head(100)) # Dibatasi 100 baris pertama agar optimal

    st.divider()

    # ==========================================
    # BAB 2.4: Menampilkan Grafik (Visualisasi Data)
    # ==========================================
    st.subheader(":material/bar_chart: Visualisasi Data")
    tab1, tab2 = st.tabs([":material/bar_chart: Bar Chart (Distribusi)", ":material/scatter_plot: Scatter Chart (Luas vs Harga)"])
    
    with tab1:
        if 'SalePrice' in df.columns:
            st.bar_chart(df_filtered['SalePrice'].head(50))
            
    with tab2:
        kolom_area = 'Lot Area' if 'Lot Area' in df.columns else df.columns[4]
        if 'SalePrice' in df.columns:
            chart_data = df_filtered[[kolom_area, 'SalePrice']].dropna()
            st.scatter_chart(data=chart_data, x=kolom_area, y='SalePrice')

except FileNotFoundError:
    # ==========================================
    # BAB 2.8: Control Flow
    # ==========================================
    st.error("File 'AmesHousing.csv' tidak ditemukan! Pastikan file dataset sudah ditaruh di folder yang sama dengan app.py.")
    st.stop()
