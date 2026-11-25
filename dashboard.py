import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. KONFIGURASI HALAMAN STREAMLIT (BAB 2.7: LAYOUT & CONTAINER)
# ==============================================================================
st.set_page_config(
    page_title="Ames Housing ML & EDA Engine",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CACHING & MEMUAT DATASET (BAB 2.3: MENAMPILKAN DATA)
# ==============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("AmesHousing.csv")
        return df
    except FileNotFoundError:
        st.error("Error: Berkas 'AmesHousing.csv' tidak ditemukan di direktori lokal.")
        st.stop()

df_raw = load_data()

# Metadata pemetaan kolom berdasarkan dokumentasi resmi Ames Housing (Dean De Cock, 2011)
COLUMN_MAPPING_METADATA = {
    "Order": {
        "makna": "Nomor urut baris observasi pada dataset",
        "peran": "Identifier",
        "kelayakan": "Tidak Layak (Identifier)",
        "tindakan": "Dikecualikan dari fitur prediktif karena hanya nomor urut administratif",
    },
    "PID": {
        "makna": "Parcel Identification Number (Nomor Pajak Properti Kota Ames)",
        "peran": "Identifier",
        "kelayakan": "Tidak Layak (Identifier Unik)",
        "tindakan": "Dihapus dari pemodelan (Uniqueness Ratio = 1.0000, memicu overfitting/noise pada pohon keputusan)",
    },
    "MS SubClass": {
        "makna": "Kode klasifikasi tipe bangunan tempat tinggal (20, 30, ..., 190)",
        "peran": "Fitur (Nominal berkode angka)",
        "kelayakan": "Layak",
        "tindakan": "Perlu diperlakukan sebagai kategori (One-Hot/Target Encoding) bila dimodelkan",
    },
    "MS Zoning": {
        "makna": "Klasifikasi zonasi umum properti (RL, RM, FV, C, dll.)",
        "peran": "Fitur Kategorikal",
        "kelayakan": "Layak",
        "tindakan": "Encoding kategorikal setelah train-test split",
    },
    "Lot Frontage": {
        "makna": "Panjang jalan yang terhubung langsung ke properti (dalam feet)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Layak",
        "tindakan": "Memiliki 490 missing values (16.72%); wajib imputasi (median/KNN) tanpa listwise deletion",
    },
    "Lot Area": {
        "makna": "Luas total kavling tanah dalam satuan square feet (sqft)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Layak",
        "tindakan": "Distribusi menceng kanan ekstrem (skewness = 12.82); gunakan transformasi log1p / RobustScaler",
    },
    "Neighborhood": {
        "makna": "Lokasi lingkungan fisik di dalam batas kota Ames, Iowa (28 wilayah)",
        "peran": "Fitur Kategorikal Spasial",
        "kelayakan": "Layak (Fitur Kunci)",
        "tindakan": "Digunakan untuk pengelompokan spasial, visualisasi grafik batang, dan pemetaan area Ames",
    },
    "Bldg Type": {
        "makna": "Tipe hunian (1Fam, 2fmCon, Duplex, Twnhs, TwnhsE)",
        "peran": "Fitur Kategorikal",
        "kelayakan": "Layak",
        "tindakan": "Membentuk subpopulasi luas bangunan yang signifikan (ANOVA p < 1e-12)",
    },
    "House Style": {
        "makna": "Gaya arsitektur tingkat bangunan (1Story, 2Story, 1.5Fin, dll.)",
        "peran": "Fitur Kategorikal",
        "kelayakan": "Layak",
        "tindakan": "Digunakan pada eksplorasi multiselect dan analisis subgrup",
    },
    "Overall Qual": {
        "makna": "Penilaian kualitas material dan penyelesaian akhir rumah (skala 1-10)",
        "peran": "Fitur Numerik Ordinal",
        "kelayakan": "Layak (Prediktor Terkuat)",
        "tindakan": "Korelasi tertinggi terhadap SalePrice (r = 0.7993); siap digunakan",
    },
    "Overall Cond": {
        "makna": "Penilaian kondisi keseluruhan rumah saat ini (skala 1-10)",
        "peran": "Fitur Numerik Ordinal",
        "kelayakan": "Layak",
        "tindakan": "Pertahankan sebagai fitur ordinal",
    },
    "Year Built": {
        "makna": "Tahun konstruksi awal bangunan",
        "peran": "Fitur Numerik Temporal",
        "kelayakan": "Layak",
        "tindakan": "Digunakan untuk analisis tren deret waktu (line chart) dan usia bangunan",
    },
    "Year Remod/Add": {
        "makna": "Tahun renovasi atau penambahan bangunan (sama dengan Year Built jika belum direnovasi)",
        "peran": "Fitur Numerik Temporal",
        "kelayakan": "Layak",
        "tindakan": "Dapat dikombinasikan dengan Year Built menjadi fitur selisih usia renovasi",
    },
    "Exter Qual": {
        "makna": "Kualitas material eksterior (Ex, Gd, TA, Fa, Po)",
        "peran": "Fitur Kategorikal Ordinal",
        "kelayakan": "Layak",
        "tindakan": "Ordinal encoding (Po=1 hingga Ex=5)",
    },
    "Total Bsmt SF": {
        "makna": "Total luas area ruang bawah tanah / basement (sqft)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Layak",
        "tindakan": "Terdapat 1 missing value; imputasi median pada pipeline",
    },
    "1st Flr SF": {
        "makna": "Luas lantai pertama (sqft)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Perlu Seleksi/Mitigasi VIF",
        "tindakan": "Memiliki multikolinearitas tinggi dengan Gr Liv Area (VIF = 74.85); gunakan PCA atau seleksi fitur",
    },
    "2nd Flr SF": {
        "makna": "Luas lantai kedua (sqft)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Perlu Seleksi/Mitigasi VIF",
        "tindakan": "Kolinear terhadap Gr Liv Area (VIF = 86.98); direduksi jika Gr Liv Area sudah dipakai",
    },
    "Gr Liv Area": {
        "makna": "Luas ruang hunian di atas permukaan tanah / Ground Living Area (sqft)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Layak (Fitur Utama)",
        "tindakan": "Cek 3 anomali penjualan parsial (Gr Liv Area > 4000 & SalePrice < $300,000)",
    },
    "Full Bath": {
        "makna": "Jumlah kamar mandi penuh di atas permukaan tanah",
        "peran": "Fitur Numerik Diskrit",
        "kelayakan": "Layak",
        "tindakan": "Siap digunakan",
    },
    "TotRms AbvGrd": {
        "makna": "Total ruangan di atas tanah (tidak termasuk kamar mandi)",
        "peran": "Fitur Numerik Diskrit",
        "kelayakan": "Layak",
        "tindakan": "Siap digunakan",
    },
    "Garage Cars": {
        "makna": "Kapasitas garasi dalam jumlah mobil",
        "peran": "Fitur Numerik Diskrit",
        "kelayakan": "Layak",
        "tindakan": "1 missing value; kolinear dengan Garage Area",
    },
    "Garage Area": {
        "makna": "Luas garasi dalam satuan square feet (sqft)",
        "peran": "Fitur Numerik Kontinu",
        "kelayakan": "Layak",
        "tindakan": "1 missing value; imputasi median",
    },
    "Pool QC": {
        "makna": "Kualitas kolam renang (Ex, Gd, TA, Fa, atau NA jika tidak ada kolam)",
        "peran": "Fitur Kategorikal",
        "kelayakan": "Layak dengan Transformasi",
        "tindakan": "2.917 nilai kosong berarti 'Tidak Ada Kolam Renang' (bukan data rusak)",
    },
    "Mo Sold": {
        "makna": "Bulan transaksi penjualan (1-12)",
        "peran": "Metadata / Fitur Siklikal",
        "kelayakan": "Layak",
        "tindakan": "Digunakan untuk eksplorasi musiman transaksi",
    },
    "Yr Sold": {
        "makna": "Tahun transaksi penjualan (2006-2010)",
        "peran": "Metadata / Fitur Temporal",
        "kelayakan": "Layak",
        "tindakan": "Digunakan untuk filter periode transaksi",
    },
    "Sale Type": {
        "makna": "Jenis skema transaksi penjualan (WD, New, COD, dll.)",
        "peran": "Metadata Transaksi",
        "kelayakan": "Hati-hati (Potensi Kebocoran Konteks)",
        "tindakan": "Periksa apakah informasi tersedia sebelum transaksi selesai",
    },
    "Sale Condition": {
        "makna": "Kondisi penjualan (Normal, Partial, Abnorml, Family, Alloca, AdjLand)",
        "peran": "Metadata Transaksi",
        "kelayakan": "Hati-hati (Identifikasi Anomali)",
        "tindakan": "Menjelaskan 3 rumah outlier ekstrem (>4000 sqft) yang berstatus 'Partial'",
    },
    "SalePrice": {
        "makna": "Harga jual akhir properti dalam satuan US Dollar ($)",
        "peran": "Target (Variabel Dependen - Regresi)",
        "kelayakan": "Target Utama",
        "tindakan": "Target kontinu (Regresi); skewness = 1.744, direkomendasikan transformasi log1p saat pemodelan linear",
    },
}

@st.cache_data
def get_column_mapping_df(df):
    rows = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing_cnt = int(df[col].isna().sum())
        missing_pct = round(missing_cnt / len(df) * 100, 2)
        unique_cnt = int(df[col].nunique(dropna=True))

        if col in COLUMN_MAPPING_METADATA:
            meta = COLUMN_MAPPING_METADATA[col]
            makna = meta["makna"]
            peran = meta["peran"]
            kelayakan = meta["kelayakan"]
            tindakan = meta["tindakan"]
        else:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            makna = f"Atribut karakteristik properti ({'numerik' if is_num else 'kategorikal'}) pada Ames Housing"
            peran = "Fitur Numerik" if is_num else "Fitur Kategorikal"
            kelayakan = "Layak"
            if missing_cnt > 0:
                tindakan = f"Imputasi nilai kosong ({missing_cnt} baris / {missing_pct}%) sesuai konteks domain"
            else:
                tindakan = "Siap digunakan setelah encoding/scaling sesuai kebutuhan algoritma"

        rows.append(
            {
                "Nama Kolom Asli": col,
                "Tipe Data": dtype_str,
                "Nilai Unik": unique_cnt,
                "Missing": f"{missing_cnt} ({missing_pct}%)",
                "Makna Kolom": makna,
                "Peran Kolom": peran,
                "Kelayakan Pakai": kelayakan,
                "Tindakan yang Diperlukan": tindakan,
            }
        )
    return pd.DataFrame(rows)

@st.cache_data
def get_dataset_structure_audit(df):
    missing_series = df.isna().sum()
    missing_pct = (missing_series / len(df) * 100).round(2)
    dtype_df = pd.DataFrame(
        {
            "Nama Kolom": df.columns,
            "Tipe Data": [str(t) for t in df.dtypes],
            "Jumlah Non-Null": df.notna().sum().values,
            "Jumlah Missing": missing_series.values,
            "Persentase Missing (%)": missing_pct.values,
            "Jumlah Nilai Unik": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )

    missing_only_df = (
        dtype_df[dtype_df["Jumlah Missing"] > 0]
        .sort_values(by="Jumlah Missing", ascending=False)
        .reset_index(drop=True)
    )

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_summary = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=True)
        top_vals = ", ".join([f"{k} ({v})" for k, v in vc.head(4).items()])
        cat_summary.append(
            {
                "Kolom Kategorikal": c,
                "Jumlah Kategori Unik": int(df[c].nunique(dropna=True)),
                "Jumlah Missing": int(df[c].isna().sum()),
                "Contoh Kategori Terbanyak": top_vals,
            }
        )
    cat_df = pd.DataFrame(cat_summary)

    num_df = df.select_dtypes(include="number")
    desc_df = num_df.describe().T
    desc_df["median"] = num_df.median()
    desc_df["skewness"] = num_df.skew()
    desc_df["kurtosis"] = num_df.kurt()
    desc_df = desc_df[
        ["count", "mean", "std", "min", "25%", "median", "75%", "max", "skewness", "kurtosis"]
    ].round(2)
    desc_df.index.name = "Kolom Numerik"

    return {
        "dtype_table": dtype_df,
        "missing_table": missing_only_df,
        "categorical_table": cat_df,
        "numeric_descriptive": desc_df.reset_index(),
    }

# ==============================================================================
# DEFINISI HALAMAN-HALAMAN MODUL
# ==============================================================================
def page_beranda():
    st.title("Dashboard Analisis Properti & Machine Learning Ames Housing")
    st.markdown(
        "Aplikasi ini mengimplementasikan **framework Streamlit** untuk mendiagnosis 12 masalah utama *data preprocessing*, "
        "melakukan *feature engineering*, reduksi dimensi, dan membandingkan **3 paradigma algoritma Machine Learning**."
    )
    st.divider()

    # BAB 2.3: METRIC DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observasi", f"{df_raw.shape[0]:,} baris")
    col2.metric("Total Fitur", f"{df_raw.shape[1]} kolom")
    col3.metric("Rata-rata Harga", f"${df_raw['SalePrice'].mean():,.2f}")
    col4.metric("Median Luas Bangunan", f"{df_raw['Gr Liv Area'].median():,.0f} sqft")

    st.divider()
    st.subheader("Preview Dataset Utama (st.dataframe)")
    st.dataframe(df_raw.head(10), use_container_width=True)


def page_tentang_dataset():
    st.title(":material/database: Tentang Dataset & Audit Kelompok (Ames Housing)")
    st.caption("Pemeriksaan struktur menyeluruh, pemetaan 82 kolom, analisis variabel target, dan identifikasi kualitas data sesuai modul Bab 1 & 2.")

    audit_struct = get_dataset_structure_audit(df_raw)

    tab_p1, tab_p2, tab_p3, tab_p4 = st.tabs(
        [
            ":material/table_chart: Pemeriksaan Struktur & Missing Values",
            ":material/map: Pemetaan Arti Kolom (82 Kolom)",
            ":material/analytics: Target & Distribusi Log1p",
            ":material/verified: Audit Kualitas & Anomali Data",
        ]
    )

    with tab_p1:
        st.subheader("Pemeriksaan Struktur & Kelengkapan Dataset")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Dimensi Data", f"{df_raw.shape[0]:,} × {df_raw.shape[1]}", "82 Kolom Total")
        m_c2.metric("Baris Duplikat", f"{df_raw.duplicated().sum()} Baris", "Bebas Duplikasi")
        m_c3.metric("Kolom dengan Missing", f"{(df_raw.isna().sum() > 0).sum()} dari 82 Kolom", f"{df_raw.isna().sum().sum():,} sel kosong")
        kelengkapan_pct = (1 - df_raw.isna().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100
        m_c4.metric("Kelengkapan Sel", f"{kelengkapan_pct:.2f}%", "Rasio Validitas")

        col_dt, col_ms = st.columns(2)
        with col_dt:
            st.markdown("#### a. Tipe Data & Nilai Unik Setiap Kolom")
            st.dataframe(audit_struct["dtype_table"], use_container_width=True, height=320)
        with col_ms:
            st.markdown("#### b. Daftar 27 Kolom yang Memiliki Nilai Kosong (Missing Values)")
            st.dataframe(audit_struct["missing_table"], use_container_width=True, height=320)

        st.markdown("#### c. Jumlah Nilai Unik pada 43 Kolom Kategorikal")
        st.dataframe(audit_struct["categorical_table"], use_container_width=True, height=280)

        st.markdown("#### d. Ringkasan Statistik Deskriptif 39 Kolom Numerik")
        st.dataframe(audit_struct["numeric_descriptive"], use_container_width=True, height=340)

    with tab_p2:
        st.subheader("Pemetaan Arti, Peran, & Tindakan Setiap Kolom")
        st.write(
            "Tabel berikut memetakan seluruh 82 kolom `AmesHousing.csv` beserta tipe data, nilai unik, "
            "persentase nilai kosong, makna domain, peran variabel, kelayakan analisis, dan tindakan preprocessing."
        )
        mapping_df = get_column_mapping_df(df_raw)
        peran_filter = st.multiselect(
            "Filter berdasarkan Peran Kolom:",
            options=mapping_df["Peran Kolom"].unique().tolist(),
            default=mapping_df["Peran Kolom"].unique().tolist(),
        )
        st.dataframe(
            mapping_df[mapping_df["Peran Kolom"].isin(peran_filter)],
            use_container_width=True,
            height=450,
        )

    with tab_p3:
        st.subheader("Penentuan Variabel Target & Karakteristik Distribusi")
        st.markdown(
            """
            - **Variabel Target Utama:** `SalePrice` (Harga jual properti dalam US Dollar, rentang **$12,789** hingga **$755,000**).
            - **Jenis Masalah Pembelajaran Mesin:** **Supervised Learning — Regresi** (target berupa nilai numerik kontinu).
            - **Karakteristik Distribusi Target:**
              - Sebelum transformasi: Rata-rata = **$180,796.06**, Median = **$160,000.00**, Skewness = **`1.744`** (menceng kanan), Kurtosis = **`5.119`** (leptokurtik).
              - Sesudah transformasi $\\log(1 + \\text{SalePrice})$: Skewness = **`-0.015`**, Kurtosis = **`1.514`** (mendekati distribusi Normal Gaussian).
            """
        )
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(df_raw["SalePrice"] / 1000, bins=35, color="#3b82f6", edgecolor="white", alpha=0.85)
        ax1.axvline(df_raw["SalePrice"].mean() / 1000, color="red", linestyle="--", label=f"Mean: ${df_raw['SalePrice'].mean():,.0f}")
        ax1.axvline(df_raw["SalePrice"].median() / 1000, color="green", linestyle="-", label=f"Median: ${df_raw['SalePrice'].median():,.0f}")
        ax1.set_title(f"Distribusi SalePrice Mentah (Skew: {df_raw['SalePrice'].skew():.3f})")
        ax1.set_xlabel("Harga Jual (Ribu USD)")
        ax1.set_ylabel("Frekuensi")
        ax1.legend()

        log_target = np.log1p(df_raw["SalePrice"])
        ax2.hist(log_target, bins=35, color="#10b981", edgecolor="white", alpha=0.85)
        ax2.axvline(log_target.mean(), color="red", linestyle="--", label=f"Mean: {log_target.mean():.2f}")
        ax2.axvline(log_target.median(), color="green", linestyle="-", label=f"Median: {log_target.median():.2f}")
        ax2.set_title(f"Distribusi log1p(SalePrice) (Skew: {log_target.skew():.3f})")
        ax2.set_xlabel("Nilai log1p(SalePrice)")
        ax2.set_ylabel("Frekuensi")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig_dist)

    with tab_p4:
        st.subheader("Audit Kualitas Data, Kolom Identifier, & Anomali")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("#### 1. Audit Kolom Identifier (`Order` & `PID`)")
            pid_unique_ratio = df_raw["PID"].nunique() / len(df_raw)
            st.write(
                f"- Rasio keunikan `PID`: **`{pid_unique_ratio:.4f}`** (100% unik per properti).\n"
                f"- **Risiko Overfitting:** Jika `PID` dimasukkan ke dalam algoritma pohon keputusan (Decision Tree), "
                f"`PID` akan mendominasi *Feature Importance* (hingga > 0.40) karena bertindak sebagai 'nomor identitas unik' "
                f"yang dihafal model, bukan prediktor fisik rumah.\n"
                f"- **Rekomendasi:** Wajib di-*drop* dari set fitur sebelum pemodelan."
            )
        with col_q2:
            st.markdown("#### 2. Deteksi 3 Anomali Penjualan Parsial (*Dean De Cock*)")
            st.write(
                "Terdapat 3 rumah dengan luas `Gr Liv Area > 4000 sqft` namun harga jual di bawah `$300,000` "
                "karena berstatus `Partial` (rumah belum selesai dibangun saat transaksi berlangsung):"
            )
            anomali_partial = df_raw[(df_raw["Gr Liv Area"] > 4000) & (df_raw["SalePrice"] < 300000)][
                ["PID", "Gr Liv Area", "Overall Qual", "Sale Condition", "SalePrice"]
            ]
            st.dataframe(anomali_partial, use_container_width=True)

# ==============================================================================
# HALAMAN 2: DIAGNOSTIC EDA & PREPROCESSING (10 MASALAH DIVERIFIKASI)
# ==============================================================================
def page_diagnostic_eda():
    st.title("Diagnosa & Solusi Preprocessing Data (Masalah 1 - 10)")
    st.caption("Eksplorasi mendalam terkait kebocoran data, skewness, imputasi, VIF, dan penanganan outlier.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Leakage & Skewness (M1-M2)",
        "Imputasi & Multikolinearitas (M3-M4)",
        "Target Log & Seleksi Fitur (M5-M6)",
        "Kardinalitas & Outlier Masking (M7-M8)",
        "Multimodal & True Signal (M9-M10)"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: MASALAH 1 & MASALAH 2
    # --------------------------------------------------------------------------
    with tab1:
        st.header("Masalah 1: Kebocoran Data (Data Leakage) saat Preprocessing")
        fitur_m1 = ["Lot Area", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Frontage"]
        X_m1 = df_raw[fitur_m1].copy()
        y_m1 = df_raw["SalePrice"]

        # Sebelum (Bocor)
        scaler_bocor = StandardScaler()
        X_bocor = scaler_bocor.fit_transform(X_m1.fillna(X_m1.median()))
        X_tr_l, X_ts_l, y_tr_l, y_ts_l = train_test_split(X_bocor, y_m1, test_size=0.2, random_state=42)
        r2_sebelum_m1 = LinearRegression().fit(X_tr_l, y_tr_l).score(X_ts_l, y_ts_l)

        # Sesudah (Isolasi Pipeline)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X_m1, y_m1, test_size=0.2, random_state=42)
        pipe_m1 = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        pipe_m1.fit(X_tr, y_tr)
        r2_sesudah_m1 = pipe_m1.score(X_ts, y_ts)

        m1_col1, m1_col2 = st.columns(2)
        m1_col1.metric("R² Data Uji (Sebelum / Leaked)", f"{r2_sebelum_m1:.4f}")
        m1_col2.metric("R² Data Uji (Sesudah / Isolated Pipeline)", f"{r2_sesudah_m1:.4f}")
        st.info("**Solusi:** `Pipeline` memastikan statistik scaler & imputer hanya dihitung dari data latih (*Train Set*).")

        st.divider()
        st.header("Masalah 2: Penskalaan Fitur Menceng Ekstrem & Evaluasi StandardScaler")
        skew_lot = df_raw["Lot Area"].skew()
        lot_std = StandardScaler().fit_transform(df_raw[["Lot Area"]])
        skew_sebelum_m2 = pd.Series(lot_std.flatten()).skew()
        lot_log = np.log1p(df_raw["Lot Area"])
        skew_sesudah_m2 = pd.Series(lot_log).skew()

        fitur_demo = ["Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Area"]
        X_demo = df_raw[fitur_demo].fillna(df_raw[fitur_demo].median())
        y_demo = df_raw["SalePrice"]
        X_tr_d, X_ts_d, y_tr_d, y_ts_d = train_test_split(X_demo, y_demo, test_size=0.2, random_state=42)
        r2_knn_raw = KNeighborsRegressor(n_neighbors=5).fit(X_tr_d, y_tr_d).score(X_ts_d, y_ts_d)
        r2_knn_scaled = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=5))]).fit(X_tr_d, y_tr_d).score(X_ts_d, y_ts_d)

        m2_col1, m2_col2 = st.columns(2)
        with m2_col1:
            st.write("**Efek Log Transform pada Skewness Lot Area:**")
            st.write(f"- Skewness Awal: `{skew_lot:.3f}`")
            st.write(f"- Skewness setelah StandardScaler: `{skew_sebelum_m2:.3f}` (Tidak Berubah)")
            st.write(f"- Skewness setelah Log1p Transform: `{skew_sesudah_m2:.3f}` (Normal)")
        with m2_col2:
            st.write("**Revaluasi StandardScaler pada K-NN:**")
            st.write(f"- K-NN Tanpa Scaler (Buta Skala): R² = `{r2_knn_raw:.4f}`")
            st.write(f"- K-NN + StandardScaler (Skala Adil): R² = `{r2_knn_scaled:.4f}`")

    # --------------------------------------------------------------------------
    # TAB 2: MASALAH 3 & MASALAH 4
    # --------------------------------------------------------------------------
    with tab2:
        st.header("Masalah 3: Penghapusan Baris Membabi Buta (Listwise Deletion)")
        df_dropped = df_raw.dropna(subset=["Lot Frontage"])
        mean_target_asli = df_raw["SalePrice"].mean()
        mean_target_dropped = df_dropped["SalePrice"].mean()

        m3_col1, m3_col2 = st.columns(2)
        m3_col1.metric("Sampel Setelah `dropna()`", f"{len(df_dropped)} baris", f"{len(df_dropped)-len(df_raw)} baris")
        m3_col2.metric("Pergeseran Rata-rata SalePrice", f"${mean_target_dropped:,.2f}", f"${mean_target_dropped - mean_target_asli:+,.2f}")
        st.success("**Solusi:** `KNNImputer(n_neighbors=5)` menjaga 100% sampel utuh tanpa bias seleksi.")

        st.divider()
        st.header("Masalah 4: Jebakan Multikolinearitas Tinggi (VIF Ekstrem)")
        def kalkulasi_vif(data):
            vif_list = []
            for col in data.columns:
                X_other = data.drop(columns=[col])
                y_col = data[col]
                r2 = LinearRegression().fit(X_other, y_col).score(X_other, y_col)
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
                vif_list.append(vif)
            return pd.DataFrame({"Fitur": data.columns, "VIF": np.round(vif_list, 2)})

        vif_sebelum = kalkulasi_vif(df_raw[["Gr Liv Area", "1st Flr SF", "2nd Flr SF", "Total Bsmt SF"]].dropna())
        vif_sesudah = kalkulasi_vif(df_raw[["Gr Liv Area", "Garage Area"]].dropna())

        m4_col1, m4_col2 = st.columns(2)
        with m4_col1:
            st.write("**VIF Sebelum Mitigasi (Multikolinearitas Eksplosif):**")
            st.table(vif_sebelum)
        with m4_col2:
            st.write("**VIF Sesudah Mitigasi (Eliminasi Redundansi):**")
            st.table(vif_sesudah)

    # --------------------------------------------------------------------------
    # TAB 3: MASALAH 5 & MASALAH 6
    # --------------------------------------------------------------------------
    with tab3:
        st.header("Masalah 5: Pelanggaran Asumsi Normalitas Target (Skewed Target)")
        skew_sebelum_m5 = df_raw["SalePrice"].skew()
        kurt_sebelum_m5 = df_raw["SalePrice"].kurt()
        log_saleprice = np.log1p(df_raw["SalePrice"])
        skew_sesudah_m5 = log_saleprice.skew()
        kurt_sesudah_m5 = log_saleprice.kurt()

        m5_col1, m5_col2 = st.columns(2)
        m5_col1.metric("SalePrice Mentah", f"Skew: {skew_sebelum_m5:.3f}", f"Kurt: {kurt_sebelum_m5:.3f}")
        m5_col2.metric("SalePrice Log1p", f"Skew: {skew_sesudah_m5:.3f}", f"Kurt: {kurt_sesudah_m5:.3f}")

        fig_m5, ax_m5 = plt.subplots(1, 2, figsize=(12, 4))
        ax_m5[0].hist(df_raw["SalePrice"], bins=30, color="crimson", alpha=0.7)
        ax_m5[0].set_title("Distribusi Target Mentah (Right-Skewed)")
        ax_m5[1].hist(log_saleprice, bins=30, color="teal", alpha=0.7)
        ax_m5[1].set_title("Distribusi Target Log1p (Gaussian)")
        st.pyplot(fig_m5)

        st.divider()
        st.header("Masalah 6: Bias Asumsi Linearitas (Pearson vs Spearman vs Mutual Info)")
        fitur_m6 = ["Overall Qual", "Gr Liv Area", "Year Built", "Total Bsmt SF", "Lot Area"]
        X_m6 = df_raw[fitur_m6].fillna(df_raw[fitur_m6].median())
        y_target_m6 = df_raw["SalePrice"]

        pearson_scores = X_m6.apply(lambda col: stats.pearsonr(col, y_target_m6)[0]).abs()
        spearman_scores = X_m6.apply(lambda col: stats.spearmanr(col, y_target_m6)[0]).abs()
        mi_scores = pd.Series(mutual_info_regression(X_m6, y_target_m6, random_state=42), index=X_m6.columns)

        komparasi_m6 = pd.DataFrame({
            "Pearson (Linear)": pearson_scores,
            "Spearman (Monotonik)": spearman_scores,
            "Mutual Info (Non-Linear)": mi_scores
        }).round(4)

        st.dataframe(komparasi_m6, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: MASALAH 7 & MASALAH 8
    # --------------------------------------------------------------------------
    with tab4:
        st.header("Masalah 7: Fitur Identitas Unik Berkardinalitas Tinggi (PID)")
        df_id = df_raw[['PID', 'Gr Liv Area', 'Overall Qual', 'SalePrice']].dropna().copy()
        tree_sebelum = DecisionTreeRegressor(random_state=42).fit(df_id[['PID', 'Gr Liv Area', 'Overall Qual']], df_id['SalePrice'])
        tree_sesudah = DecisionTreeRegressor(random_state=42).fit(df_id[['Gr Liv Area', 'Overall Qual']], df_id['SalePrice'])

        m7_col1, m7_col2 = st.columns(2)
        with m7_col1:
            st.write("**Importance Sebelum Drop `PID` (Tercemar Identitas):**")
            st.dataframe(pd.Series(tree_sebelum.feature_importances_, index=['PID', 'Gr Liv Area', 'Overall Qual']), use_container_width=True)
        with m7_col2:
            st.write("**Importance Sesudah Drop `PID` (Murni Fisik):**")
            st.dataframe(pd.Series(tree_sesudah.feature_importances_, index=['Gr Liv Area', 'Overall Qual']), use_container_width=True)

        st.divider()
        st.header("Masalah 8: Masking Effect pada Deteksi Outlier Parametrik (Z-Score)")
        lot_area_sample = df_raw['Lot Area'].dropna().copy()
        lot_area_masked = pd.concat([lot_area_sample, pd.Series([500000.0])]).reset_index(drop=True)
        z_scores = np.abs(stats.zscore(lot_area_masked))
        outliers_sebelum_m8 = lot_area_masked[z_scores > 3]

        med_m8 = np.median(lot_area_masked)
        mad_m8 = np.median(np.abs(lot_area_masked - med_m8))
        modified_z = 0.6745 * np.abs(lot_area_masked - med_m8) / mad_m8
        outliers_sesudah_m8 = lot_area_masked[modified_z > 3.5]

        m8_col1, m8_col2 = st.columns(2)
        m8_col1.metric("Z-Score Klasik (Ter-masking)", f"{len(outliers_sebelum_m8)} outlier")
        m8_col2.metric("Modified Z-Score (Robust MAD)", f"{len(outliers_sesudah_m8)} outlier")

    # --------------------------------------------------------------------------
    # TAB 5: MASALAH 9 & MASALAH 10
    # --------------------------------------------------------------------------
    with tab5:
        st.header("Masalah 9: Pengabaian Multimodalitas Fitur (Subpopulasi Tersembunyi)")
        mean_global_m9 = df_raw['Gr Liv Area'].mean()
        mean_per_grup_m9 = df_raw.groupby('Bldg Type')['Gr Liv Area'].mean()
        grup_data_m9 = [group['Gr Liv Area'].dropna() for name, group in df_raw.groupby('Bldg Type')]
        f_stat_m9, p_val_m9 = stats.f_oneway(*grup_data_m9)

        st.write(f"**Mean Global `Gr Liv Area`:** `{mean_global_m9:.2f} sqft`")
        st.write("**Mean Berdasarkan `Bldg Type` (ANOVA p-value = {:.5e}):**".format(p_val_m9))
        st.dataframe(mean_per_grup_m9.round(2), use_container_width=True)

        st.divider()
        st.header("Masalah 10: Penanganan Outlier Sejati (Dean De Cock) vs True Signal Rumah Mewah")
        outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
        df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)

        outliers_df_m10 = df_clean[df_clean['Gr Liv Area'] > (df_clean['Gr Liv Area'].quantile(0.75) + 1.5 * (df_clean['Gr Liv Area'].quantile(0.75) - df_clean['Gr Liv Area'].quantile(0.25)))]

        m10_col1, m10_col2, m10_col3 = st.columns(3)
        m10_col1.metric("Anomali Dipangkas (De Cock)", f"{outlier_anomali_mask.sum()} baris")
        m10_col2.metric("Mean Harga Outlier Mewah", f"${outliers_df_m10['SalePrice'].mean():,.2f}")
        m10_col3.metric("Mean Harga Seluruh Populasi", f"${df_clean['SalePrice'].mean():,.2f}")

# ==============================================================================
# HALAMAN 3: REDUKSI DIMENSI BERBASIS PROYEKSI ORTOGONAL (PCA)
# ==============================================================================
def page_pca():
    st.title("Reduksi Dimensi Proyeksi Ortogonal (PCA)")
    st.markdown("Mengompresi fitur berkorelasi tinggi menjadi komponen utama yang saling **ortogonal (korelasi r ≈ 0)**.")

    outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
    df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)

    fitur_pca_list = ["Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Area", "1st Flr SF"]
    df_pca = df_clean[fitur_pca_list].dropna().copy()
    X_train_pca, X_test_pca = train_test_split(df_pca, test_size=0.2, random_state=42)

    pipe_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42))
    ])
    X_train_trans = pipe_pca.fit_transform(X_train_pca)
    pca_engine = pipe_pca.named_steps["pca"]
    ind_var = pca_engine.explained_variance_ratio_
    cum_var = np.cumsum(ind_var)

    col_pca1, col_pca2 = st.columns(2)
    with col_pca1:
        st.metric("Komponen Terpilih (Cutoff ≥ 95%)", f"{pca_engine.n_components_} Components")
        st.metric("Total Kumulatif Varians Terjaga", f"{cum_var[-1] * 100:.2f}%")
    with col_pca2:
        st.write("**Explained Variance Ratio per Komponen:**")
        st.dataframe(pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(ind_var))], "Variance Ratio": ind_var.round(4)}))

    # BAB 2.4: VISUALISASI SCREE PLOT
    st.subheader("Scree Plot Kompresi PCA")
    fig_scree, ax_scree = plt.subplots(figsize=(8, 4))
    pc_range = range(1, len(ind_var) + 1)
    ax_scree.bar(pc_range, ind_var, alpha=0.5, label='Individual Var', color='navy')
    ax_scree.step(pc_range, cum_var, where='mid', label='Cumulative Var', color='crimson', linewidth=2)
    ax_scree.axhline(y=0.95, color='black', linestyle='--', label='95% Cutoff Threshold')
    ax_scree.set_xlabel('Indeks Komponen Utama (PC)')
    ax_scree.set_ylabel('Explained Variance Ratio')
    ax_scree.legend()
    ax_scree.grid(axis='y', linestyle=':')
    st.pyplot(fig_scree)

# ==============================================================================
# HALAMAN 4: KOMPARASI 3 PARADIGMA ML
# ==============================================================================
def page_komparasi_ml():
    st.title("Komparasi 3 Paradigma Algoritma Supervised Learning")
    st.markdown("Evaluasi performa model Parametrik Linear vs Berbasis Jarak (K-NN) vs Berbasis Pohon (Decision Tree & Random Forest).")

    outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
    df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)

    fitur_numerik = ["Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF", "Year Built", "Lot Area"]
    X = df_clean[fitur_numerik]
    y = df_clean["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fitur_skewed = ["Lot Area", "Gr Liv Area", "Total Bsmt SF"]
    fitur_reguler = [f for f in fitur_numerik if f not in fitur_skewed]

    preprocessor_linear_knn = ColumnTransformer(
        transformers=[
            ("skew_transformer", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("yeo_johnson", PowerTransformer(method="yeo-johnson"))
            ]), fitur_skewed),
            ("reguler_scaler", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), fitur_reguler)
        ]
    )
    preprocessor_tree = SimpleImputer(strategy="median")

    # Construct Models
    model_linear = TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor_linear_knn), ("reg", LinearRegression())]),
        func=np.log1p, inverse_func=np.expm1
    )
    model_knn = TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor_linear_knn), ("reg", KNeighborsRegressor(n_neighbors=5))]),
        func=np.log1p, inverse_func=np.expm1
    )
    model_tree = Pipeline([("prep", preprocessor_tree), ("reg", DecisionTreeRegressor(max_depth=10, random_state=42))])
    model_rf = Pipeline([("prep", preprocessor_tree), ("reg", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))])

    daftar_paradigma = {
        "1. Parametrik Linear (Linear Regression)": model_linear,
        "2. Berbasis Jarak (K-Nearest Neighbors)": model_knn,
        "3. Berbasis Pohon (Decision Tree)": model_tree,
        "4. Tree Ensemble (Random Forest)": model_rf
    }

    hasil_eval = []
    for nama, mdl in daftar_paradigma.items():
        mdl.fit(X_train, y_train)
        y_tr_pred = mdl.predict(X_train)
        y_ts_pred = mdl.predict(X_test)
        hasil_eval.append({
            "Paradigma / Algoritma": nama,
            "R2 Train": round(r2_score(y_train, y_tr_pred), 4),
            "R2 Test": round(r2_score(y_test, y_ts_pred), 4),
            "RMSE ($)": round(root_mean_squared_error(y_test, y_ts_pred), 2),
            "MAE ($)": round(mean_absolute_error(y_test, y_ts_pred), 2)
        })

    df_hasil_eval = pd.DataFrame(hasil_eval)

    # BAB 2.3: MENAMPILKAN TABEL EVALUASI
    st.subheader("Tabel Ringkasan Performa Model")
    st.dataframe(df_hasil_eval, use_container_width=True)

    # BAB 2.4: MENAMPILKAN GRAFIK MATPLOTLIB/BAR CHART
    st.subheader("Visualisasi Perbandingan Akurasi (R²) & Galat (RMSE)")
    fig_eval, axes_eval = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(len(df_hasil_eval))
    bar_w = 0.35
    axes_eval[0].bar(x_pos - bar_w/2, df_hasil_eval['R2 Train'], width=bar_w, label='Train R²', color='#2980b9', alpha=0.85)
    axes_eval[0].bar(x_pos + bar_w/2, df_hasil_eval['R2 Test'], width=bar_w, label='Test R²', color='#27ae60', alpha=0.85)
    axes_eval[0].set_xticks(x_pos)
    axes_eval[0].set_xticklabels(['Linear Reg', 'K-NN', 'Decision Tree', 'Random Forest'], rotation=15, fontweight='bold')
    axes_eval[0].set_ylabel('R² Score')
    axes_eval[0].set_title('Komparasi Akurasi (R²: Train vs Test)', fontweight='bold')
    axes_eval[0].set_ylim(0.70, 1.02)
    axes_eval[0].legend(loc='lower right')
    axes_eval[0].grid(axis='y', linestyle=':')

    axes_eval[1].bar(['Linear Reg', 'K-NN', 'Decision Tree', 'Random Forest'], df_hasil_eval['RMSE ($)'], color=['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'], alpha=0.85)
    axes_eval[1].set_ylabel('RMSE ($)')
    axes_eval[1].set_title('Komparasi Galat Rata-rata (RMSE $)', fontweight='bold')
    axes_eval[1].grid(axis='y', linestyle=':')

    st.pyplot(fig_eval)

# ==============================================================================
# HALAMAN 5: SIMULATOR PREDIKSI HARGA REAL-TIME (BAB 2.5 & BAB 2.8)
# ==============================================================================
def page_simulator():
    st.title("Simulator Valuasi Real-Time Harga Rumah Ames")
    st.markdown("Masukkan karakteristik rumah untuk mendapatkan estimasi harga seketika dari model terlatih.")

    # Train Model Ensemble Random Forest untuk Backend Simulator
    outlier_anomali_mask = (df_raw['Gr Liv Area'] > 4000) & (df_raw['SalePrice'] < 300000)
    df_clean = df_raw[~outlier_anomali_mask].copy().reset_index(drop=True)
    fitur_numerik = ["Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area", "1st Flr SF", "Year Built", "Lot Area"]
    X = df_clean[fitur_numerik]
    y = df_clean["SalePrice"]

    model_rf_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("reg", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])
    model_rf_final.fit(X, y)

    # BAB 2.8: CONTROL FLOW MENGGUNAKAN FORM
    with st.form(key="form_simulasi_harga"):
        st.subheader("Input Spesifikasi Properti")
        col_in1, col_in2 = st.columns(2)

        with col_in1:
            overall_qual = st.slider("Skala Kualitas Keseluruhan (Overall Qual 1-10):", 1, 10, 7)
            gr_liv_area = st.number_input("Luas Area Tinggal Atas Tanah (Gr Liv Area sqft):", min_value=300, max_value=5000, value=1500, step=50)
            total_bsmt_sf = st.number_input("Luas Basement (Total Bsmt SF sqft):", min_value=0, max_value=3000, value=900, step=50)
            first_flr_sf = st.number_input("Luas Lantai Pertama (1st Flr SF sqft):", min_value=300, max_value=3000, value=1000, step=50)

        with col_in2:
            garage_area = st.number_input("Luas Garasi (Garage Area sqft):", min_value=0, max_value=1500, value=480, step=20)
            year_built = st.slider("Tahun Konstruksi Bangunan (Year Built):", 1872, 2010, 2003)
            lot_area = st.number_input("Luas Tanah (Lot Area sqft):", min_value=1000, max_value=100000, value=9500, step=100)

        # BAB 2.8: FORM SUBMIT BUTTON
        submit_simulasi = st.form_submit_button(label="Jalankan Estimasi Harga Properti")

    if submit_simulasi:
        data_input = pd.DataFrame([{
            "Overall Qual": overall_qual,
            "Gr Liv Area": gr_liv_area,
            "Total Bsmt SF": total_bsmt_sf,
            "Garage Area": garage_area,
            "1st Flr SF": first_flr_sf,
            "Year Built": year_built,
            "Lot Area": lot_area
        }])

        prediksi_val = model_rf_final.predict(data_input)[0]

        st.success(f"### Estimasi Valuasi Harga Jual Rumah: ${prediksi_val:,.2f}")

        # Parameter Ringkasan
        st.subheader("Ringkasan Spesifikasi Input:")
        st.dataframe(data_input, use_container_width=True)


# ==============================================================================
# INISIALISASI NAVIGASI STREAMLIT (NATIVE ST.NAVIGATION)
# ==============================================================================
nav_pages = {
    "Pengantar": [
        st.Page(page_beranda, title="Beranda", icon=":material/home:", default=True),
        st.Page(page_tentang_dataset, title="Tentang Dataset", icon=":material/database:"),
        st.Page(page_diagnostic_eda, title="Eksplorasi Data (EDA)", icon=":material/bar_chart:"),
    ],
    "Pengolahan & Model": [
        st.Page(page_pca, title="Reduksi Dimensi (PCA)", icon=":material/scatter_plot:"),
        st.Page(page_komparasi_ml, title="Komparasi 3 Paradigma ML", icon=":material/smart_toy:"),
    ],
    "Penerapan": [
        st.Page(page_simulator, title="Simulator Prediksi Harga", icon=":material/tune:"),
    ],
}

pg = st.navigation(nav_pages)
pg.run()
