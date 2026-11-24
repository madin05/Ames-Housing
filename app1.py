import numpy as np
import pandas as pd
from scipy import stats
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
import streamlit as st

# ==========================================
# BAB 2.7: Menampilkan Layout Dasar
# ==========================================
st.set_page_config(
    page_title="Ames Housing ML & Analytics App",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling navigasi radio modern seperti pill / icon bersih
st.markdown("""
<style>
/* Hilangkan lingkaran radio button di sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
/* Style item radio seperti tombol navigasi modern */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
    padding: 8px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 3px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background-color: #383838 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# BAB 2.2: Elemen Text
# ==========================================
st.title(":material/home: Aplikasi Analisis Data & Machine Learning Ames Housing")
st.markdown("**Tugas Praktikum Machine Learning - Integrasi Pipeline & Streamlit**")
st.caption(
    "Implementasi Preprocessing, Reduksi Dimensi (PCA), & Komparasi Algoritma"
)
st.divider()


# ==========================================
# Load Data & Pipeline ML (Cached)
# ==========================================
@st.cache_data
def load_and_process_data():
  # Memuat dataset utama
  df = pd.read_csv("AmesHousing.csv")

  # 1. Penanganan Outlier Sejati (Dean De Cock)
  outlier_anomali_mask = (df["Gr Liv Area"] > 4000) & (
      df["SalePrice"] < 300000
  )
  df_clean = df[~outlier_anomali_mask].copy().reset_index(drop=True)

  return df, df_clean


try:
  df, df_clean = load_and_process_data()

  # ==========================================
  # BAB 2.5 & 2.7: Layout Sidebar & Input Form
  # ==========================================
  with st.sidebar:
    st.header(":material/tune: Kontrol & Filter")

    menu_nav = st.radio(
        "Pilih Menu Utama:",
        [
            ":material/bar_chart: Eksplorasi & Statistik Data",
            ":material/search: Analisis Preprocessing & Masalah",
            ":material/smart_toy: Pemodelan & Komparasi ML",
        ],
    )

    st.divider()
    tampilkan_tabel = st.checkbox("Tampilkan Dataframe Utama", value=False)

    kolom_kategori = (
        "Neighborhood" if "Neighborhood" in df.columns else df.columns[1]
    )
    pilihan_kategori = ["Semua"] + list(df[kolom_kategori].dropna().unique())
    filter_kategori = st.selectbox(f"Pilih {kolom_kategori}", pilihan_kategori)

    if "SalePrice" in df.columns:
      min_price = int(df["SalePrice"].min())
      max_price = int(df["SalePrice"].max())
      filter_harga = st.slider(
          "Rentang SalePrice ($)",
          min_value=min_price,
          max_value=max_price,
          value=(min_price, max_price),
      )
    else:
      filter_harga = None

  # Logika Filtering Data
  df_filtered = df.copy()
  if filter_kategori != "Semua":
    df_filtered = df_filtered[
        df_filtered[kolom_kategori] == filter_kategori
    ]
  if filter_harga and "SalePrice" in df.columns:
    df_filtered = df_filtered[
        (df_filtered["SalePrice"] >= filter_harga[0])
        & (df_filtered["SalePrice"] <= filter_harga[1])
    ]

  # ==========================================
  # MENU 1: EKSPLORASI & STATISTIK DATA
  # ==========================================
  if "Eksplorasi & Statistik Data" in menu_nav:
    st.subheader(":material/analytics: Ringkasan Metrik Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Data (Baris Terfilter)", value=len(df_filtered))

    if "SalePrice" in df.columns:
      rata_harga = df_filtered["SalePrice"].mean()
      col2.metric(label="Rata-rata Harga", value=f"${rata_harga:,.0f}")
      max_harga = df_filtered["SalePrice"].max()
      col3.metric(label="Harga Tertinggi", value=f"${max_harga:,.0f}")

    if tampilkan_tabel:
      st.subheader(":material/table_chart: Preview Dataset Ames Housing")
      st.dataframe(
          df_filtered.head(100)
      )  # Dibatasi 100 baris agar performa optimal

    st.divider()
    st.subheader(":material/monitoring: Visualisasi Distribusi & Hubungan")
    tab1, tab2 = st.tabs(
        [":material/bar_chart: Bar Chart (Distribusi Harga)", ":material/scatter_plot: Scatter Chart (Luas vs Harga)"]
    )

    with tab1:
      if "SalePrice" in df.columns:
        st.bar_chart(df_filtered["SalePrice"].head(50))

    with tab2:
      kolom_area = "Lot Area" if "Lot Area" in df.columns else df.columns[4]
      if "SalePrice" in df.columns:
        chart_data = df_filtered[[kolom_area, "SalePrice"]].dropna()
        st.scatter_chart(data=chart_data, x=kolom_area, y="SalePrice")

  # ==========================================
  # MENU 2: ANALISIS PREPROCESSING & MASALAH
  # ==========================================
  elif "Analisis Preprocessing & Masalah" in menu_nav:
    st.subheader(
        ":material/build: Evaluasi Masalah Preprocessing & Mitigasi (Masalah 1 - 11)"
    )

    sub_tab = st.selectbox(
        "Pilih Topik Analisis Preprocessing:",
        [
            "1. Data Leakage & StandardScaler",
            "2. Multikolinearitas & VIF",
            "3. Outlier & Modified Z-Score",
            "4. Reduksi Dimensi (PCA)",
        ],
    )

    if sub_tab == "1. Data Leakage & StandardScaler":
      st.markdown(
          "### :material/push_pin: Masalah Kebocoran Data (Data Leakage) & Penskalaan Fitur"
      )
      st.info(
          "Melakukan fitting scaler sebelum pembagian data train-test"
          " menyebabkan kebocoran informasi. Pipeline scikit-learn"
          " mengisolasinya dengan benar."
      )

      fitur_m1 = [
          "Lot Area",
          "Gr Liv Area",
          "Total Bsmt SF",
          "Garage Area",
          "Lot Frontage",
      ]
      X_m1 = df[fitur_m1].copy()
      y_m1 = df["SalePrice"]

      # Isolasi data uji via Pipeline
      X_tr, X_ts, y_tr, y_ts = train_test_split(
          X_m1, y_m1, test_size=0.2, random_state=42
      )
      pipe_m1 = Pipeline([
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
          ("model", LinearRegression()),
      ])
      pipe_m1.fit(X_tr, y_tr)
      r2_sesudah_m1 = pipe_m1.score(X_ts, y_ts)

      skew_lot = df["Lot Area"].skew()
      lot_log = np.log1p(df["Lot Area"])
      skew_sesudah_m2 = pd.Series(lot_log).skew()

      col_a, col_b = st.columns(2)
      col_a.metric("R^2 Score Model (Clean Pipeline)", f"{r2_sesudah_m1:.4f}")
      col_b.metric(
          "Skewness Lot Area (Awal vs Log)",
          f"{skew_lot:.2f} -> {skew_sesudah_m2:.2f}",
      )

    elif sub_tab == "2. Multikolinearitas & VIF":
      st.markdown(
          "### :material/push_pin: Jebakan Multikolinearitas Tinggi & Redundansi Fitur Luas"
      )
      st.warning(
          "Fitur seperti 1st Flr SF dan 2nd Flr SF menyumbang VIF ekstrem (>70)"
          " karena redundansi terhadap Gr Liv Area."
      )

      def kalkulasi_vif(data):
        vif_list = []
        for col in data.columns:
          X_other = data.drop(columns=[col])
          y_col = data[col]
          r2 = LinearRegression().fit(X_other, y_col).score(X_other, y_col)
          vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
          vif_list.append(vif)
        return pd.DataFrame(
            {"Fitur": data.columns, "VIF": np.round(vif_list, 2)}
        )

      fitur_vif_sebelum = ["Gr Liv Area", "1st Flr SF", "2nd Flr SF"]
      df_vif = kalkulasi_vif(df[fitur_vif_sebelum].dropna())
      st.dataframe(df_vif, use_container_width=True)

    elif sub_tab == "3. Outlier & Modified Z-Score":
      st.markdown("### :material/push_pin: Masking Effect pada Deteksi Outlier")
      st.write(
          "Z-score klasik gagal mendeteksi outlier ekstrem karena nilainya"
          " sendiri menggeser mean dan standard deviasi (Masking Effect)."
          " Penggunaan **Modified Z-Score berbasis Median Absolute Deviation"
          " (MAD)** mengatasi hal ini."
      )
      lot_area_sample = df["Lot Area"].dropna().copy()
      lot_area_masked = pd.concat(
          [lot_area_sample, pd.Series([500000.0])]
      ).reset_index(drop=True)

      med_m8 = np.median(lot_area_masked)
      mad_m8 = np.median(np.abs(lot_area_masked - med_m8))
      modified_z = 0.6745 * np.abs(lot_area_masked - med_m8) / mad_m8
      outliers_sesudah_m8 = lot_area_masked[modified_z > 3.5]

      st.success(
          f"Pencilan terdeteksi menggunakan Modified Z-Score: **{len(outliers_sesudah_m8)} data**"
          " teridentifikasi akurat tanpa ter-masking."
      )

    elif sub_tab == "4. Reduksi Dimensi (PCA)":
      st.markdown(
          "### :material/push_pin: Reduksi Dimensi Berbasis Proyeksi Ortogonal (PCA)"
      )
      fitur_pca_list = [
          "Gr Liv Area",
          "Total Bsmt SF",
          "Garage Area",
          "Lot Area",
          "1st Flr SF",
      ]
      df_pca = df_clean[fitur_pca_list].dropna().copy()

      X_train_pca, X_test_pca = train_test_split(
          df_pca, test_size=0.2, random_state=42
      )
      pipe_pca = Pipeline([
          ("scaler", StandardScaler()),
          ("pca", PCA(n_components=0.95, random_state=42)),
      ])
      X_train_trans = pipe_pca.fit_transform(X_train_pca)
      pca_engine = pipe_pca.named_steps["pca"]
      ind_var = pca_engine.explained_variance_ratio_
      cum_var = np.cumsum(ind_var)

      st.write(
          f"• **Jumlah Komponen Terpilih (Cutoff >= 95% variansi):**"
          f" {pca_engine.n_components_} Komponen Utama"
      )
      st.write(
          f"• **Total Kumulatif Varians Terjaga:** {cum_var[-1] * 100:.2f}%"
      )

      # Scree Plot Streamlit
      chart_pca_df = pd.DataFrame(
          {
              "Indeks PC": [f"PC {i+1}" for i in range(len(ind_var))],
              "Explained Variance": ind_var,
          }
      )
      st.bar_chart(chart_pca_df.set_index("Indeks PC"))

  # ==========================================
  # MENU 3: PEMODELAN & KOMPARASI ML
  # ==========================================
  elif "Pemodelan & Komparasi ML" in menu_nav:
    st.subheader(
        ":material/smart_toy: Komparasi 4 Paradigma Algoritma Supervised Learning (Masalah 12)"
    )
    st.write(
        "Melatih model regresi dengan target `SalePrice` menggunakan"
        " transformasi `Log1p` untuk memenuhi asumsi normalitas."
    )

    if st.button(":material/rocket_launch: Jalankan Pelatihan Model Machine Learning"):
      with st.spinner(
          "Sedang memproses pipeline model (Linear, K-NN, Decision Tree,"
          " Random Forest)..."
      ):
        fitur_numerik = [
            "Overall Qual",
            "Gr Liv Area",
            "Total Bsmt SF",
            "Garage Area",
            "1st Flr SF",
            "Year Built",
            "Lot Area",
        ]
        X = df_clean[fitur_numerik]
        y = df_clean["SalePrice"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        fitur_skewed = ["Lot Area", "Gr Liv Area", "Total Bsmt SF"]
        fitur_reguler = [f for f in fitur_numerik if f not in fitur_skewed]

        preprocessor_linear_knn = ColumnTransformer(
            transformers=[
                (
                    "skew_transformer",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "yeo_johnson",
                            PowerTransformer(method="yeo-johnson"),
                        ),
                    ]),
                    fitur_skewed,
                ),
                (
                    "reguler_scaler",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    fitur_reguler,
                ),
            ]
        )
        preprocessor_tree = SimpleImputer(strategy="median")

        model_linear = TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", preprocessor_linear_knn),
                ("reg", LinearRegression()),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        )

        model_knn = TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", preprocessor_linear_knn),
                ("reg", KNeighborsRegressor(n_neighbors=5)),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        )

        model_tree = Pipeline([
            ("prep", preprocessor_tree),
            ("reg", DecisionTreeRegressor(max_depth=10, random_state=42)),
        ])

        model_rf = Pipeline([
            ("prep", preprocessor_tree),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                ),
            ),
        ])

        daftar_paradigma = {
            "1. Parametrik Linear (Linear Regression)": model_linear,
            "2. Berbasis Jarak (K-Nearest Neighbors)": model_knn,
            "3. Berbasis Pohon (Decision Tree)": model_tree,
            "4. Tree Ensemble (Random Forest)": model_rf,
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
              "MAE ($)": round(mean_absolute_error(y_test, y_ts_pred), 2),
          })

        df_hasil_eval = pd.DataFrame(hasil_eval)
        st.success("Pelatihan dan Evaluasi Model Selesai!")
        st.dataframe(df_hasil_eval, use_container_width=True)

        st.subheader(":material/show_chart: Komparasi Metrik R^2 Test")
        st.bar_chart(
            df_hasil_eval.set_index("Paradigma / Algoritma")["R2 Test"]
        )
    else:
      st.info(
          "Klik tombol di atas untuk melatih model dan melihat perbandingan"
          " performa secara interaktif."
      )

except FileNotFoundError:
  # ==========================================
  # BAB 2.8: Control Flow (Error Handling)
  # ==========================================
  st.error(
      "File 'AmesHousing.csv' tidak ditemukan! Pastikan file dataset sudah"
      " ditaruh di folder yang sama dengan app.py."
  )
  st.stop()