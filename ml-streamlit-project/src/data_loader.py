import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATHS = [
    PROJECT_ROOT / "data" / "AmesHousing.csv",
    PROJECT_ROOT.parent / "AmesHousing.csv",
]


# Pemetaan kolom utama berdasarkan dokumentasi resmi Ames Housing (Dean De Cock, 2011)
COLUMN_MAPPING_METADATA: Dict[str, Dict[str, str]] = {
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


def resolve_dataset_path(custom_path: Optional[str] = None) -> Path:
    """Menentukan path dataset kelompok AmesHousing.csv yang tersedia dan valid."""
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"File dataset pada path '{custom_path}' tidak ditemukan.")

    for candidate in DEFAULT_DATASET_PATHS:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "File dataset kelompok 'AmesHousing.csv' tidak ditemukan di folder 'data/' maupun root workspace."
    )


def load_group_dataset(custom_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Memuat dataset kelompok dengan validasi penuh sesuai Bagian 4.1 dan 8.1 skill.md.
    Mengembalikan (DataFrame, dictionary_info_file).
    """
    path = resolve_dataset_path(custom_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Format file '{path.suffix}' tidak didukung. Harap gunakan file CSV (.csv).")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset kosong (0 baris). Tidak dapat melanjutkan analisis.")

    required_cols = ["Order", "PID", "Gr Liv Area", "Overall Qual", "Neighborhood", "SalePrice"]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise ValueError(f"Kolom wajib dataset kelompok tidak ditemukan: {missing_req}")

    file_info = {
        "nama_file": path.name,
        "lokasi_file": str(path),
        "format_file": "CSV (Comma-Separated Values)",
        "ukuran_bytes": path.stat().st_size,
        "ukuran_kb": round(path.stat().st_size / 1024, 2),
        "jumlah_baris": int(df.shape[0]),
        "jumlah_kolom": int(df.shape[1]),
        "jumlah_duplikat": int(df.duplicated().sum()),
        "total_sel_kosong": int(df.isna().sum().sum()),
        "jumlah_kolom_kosong": int((df.isna().sum() > 0).sum()),
        "target_utama": "SalePrice (Regresi - Numerik Kontinu dalam USD)",
    }
    return df, file_info


def validate_uploaded_csv(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Memvalidasi file unggahan pengguna pada st.file_uploader() sesuai Bagian 7.3 & 8.2 skill.md.
    """
    if uploaded_file is None:
        return None, "Belum ada file yang diunggah."

    filename = getattr(uploaded_file, "name", "")
    if not filename.lower().endswith(".csv"):
        return None, f"Format file '{filename}' ditolak. Hanya file berformat .csv yang didukung."

    try:
        uploaded_file.seek(0)
        df_up = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"File CSV gagal dibaca atau rusak: {exc}"

    if df_up.empty or df_up.shape[0] == 0:
        return None, f"File '{filename}' kosong (0 baris)."

    return df_up, f"File '{filename}' valid ({df_up.shape[0]:,} baris x {df_up.shape[1]} kolom)."


def build_column_mapping_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun tabel pemetaan arti dan peran seluruh 82 kolom dataset sesuai Bagian 4.2 skill.md.
    """
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


def audit_dataset_structure(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Melakukan pemeriksaan struktur menyeluruh sesuai Bagian 4.1 skill.md:
    - tipe data setiap kolom
    - jumlah nilai kosong per kolom
    - jumlah nilai unik pada kolom kategorikal
    - statistik deskriptif kolom numerik
    """
    # 1. Tabel tipe data & missing values seluruh kolom
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

    # 2. Kolom yang memiliki missing values saja
    missing_only_df = (
        dtype_df[dtype_df["Jumlah Missing"] > 0]
        .sort_values(by="Jumlah Missing", ascending=False)
        .reset_index(drop=True)
    )

    # 3. Kolom kategorikal & nilai uniknya
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

    # 4. Statistik deskriptif kolom numerik
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

