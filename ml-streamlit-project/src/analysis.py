import math
import struct
import wave
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# Koordinat pendekatan untuk Neighborhood utama di Kota Ames, Iowa (Story County, pusat: 42.0308, -93.6319)
AMES_NEIGHBORHOOD_COORDS: Dict[str, Tuple[float, float]] = {
    "NAmes": (42.0425, -93.6135),
    "CollgCr": (42.0185, -93.6855),
    "OldTown": (42.0285, -93.6120),
    "Edwards": (42.0210, -93.6680),
    "Somerst": (42.0510, -93.6435),
    "NridgHt": (42.0615, -93.6540),
    "Gilbert": (42.0600, -93.6350),
    "Sawyer": (42.0335, -93.6710),
    "NWAmes": (42.0460, -93.6495),
    "SawyerW": (42.0350, -93.6860),
    "Mitchel": (41.9920, -93.6025),
    "BrkSide": (42.0310, -93.6265),
    "Crawfor": (42.0160, -93.6380),
    "IDOTRR": (42.0220, -93.6215),
    "Timber": (41.9985, -93.6510),
    "NoRidge": (42.0535, -93.6565),
    "StoneBr": (42.0620, -93.6320),
    "SWISU": (42.0190, -93.6530),
    "ClearCr": (42.0385, -93.6615),
    "MeadowV": (41.9945, -93.6085),
    "BrDale": (42.0530, -93.6285),
    "Blmngtn": (42.0605, -93.6410),
    "Veenker": (42.0430, -93.6540),
    "NPkVill": (42.0515, -93.6260),
    "Blueste": (42.0115, -93.6460),
    "Greens": (42.0445, -93.6485),
    "GrnHill": (42.0020, -93.6435),
    "Landmrk": (42.0475, -93.6330),
}


def get_key_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Menghitung metrik kunci dataset AmesHousing.csv untuk st.metric() (Bab 2.3.c)."""
    mean_price = float(df["SalePrice"].mean())
    median_price = float(df["SalePrice"].median())
    delta_price = mean_price - median_price

    mean_area = float(df["Gr Liv Area"].mean())
    median_area = float(df["Gr Liv Area"].median())
    delta_area = mean_area - median_area

    mean_qual = float(df["Overall Qual"].mean())
    median_qual = float(df["Overall Qual"].median())
    delta_qual = mean_qual - median_qual

    return {
        "mean_price": f"${mean_price:,.0f}",
        "delta_price": f"${delta_price:+,.0f} vs Median (${median_price:,.0f})",
        "mean_area": f"{mean_area:,.1f} sqft",
        "delta_area": f"{delta_area:+,.1f} sqft vs Median",
        "mean_qual": f"{mean_qual:.2f} / 10",
        "delta_qual": f"{delta_qual:+.2f} vs Median ({median_qual:.0f})",
    }


def prepare_chart_datasets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Menyiapkan DataFrame agregasi dari AmesHousing.csv untuk praktik grafik Bab 2.4:
    - line_chart: Tren rata-rata luas bangunan per dekade/tahun (Year Built >= 1950)
    - area_chart: Komposisi luas bangunan per tingkat kualitas (Overall Qual 1-10)
    - bar_chart: Rata-rata komponen luas per tipe bangunan (Bldg Type)
    - altair_df: Sampel terstruktur properti untuk scatter interaktif Altair & Plotly
    - map_df: Koordinat properti Ames, Iowa untuk st.map()
    """
    # 1. Data Line Chart: Rata-rata luas (Gr Liv Area, Total Bsmt SF, 1st Flr SF) per Tahun (1990-2010)
    df_recent = df[df["Year Built"] >= 1990].copy()
    line_df = (
        df_recent.groupby("Year Built")[["Gr Liv Area", "Total Bsmt SF", "1st Flr SF"]]
        .mean()
        .round(1)
    )

    # 2. Data Area Chart: Rata-rata komponen luas berdasarkan Overall Qual (1 s.d. 10)
    area_df = (
        df.groupby("Overall Qual")[["Gr Liv Area", "Total Bsmt SF", "Garage Area"]]
        .mean()
        .round(1)
    )

    # 3. Data Bar Chart: Rata-rata komponen luas berdasarkan Bldg Type
    bar_df = (
        df.groupby("Bldg Type")[["1st Flr SF", "2nd Flr SF", "Total Bsmt SF", "Garage Area"]]
        .mean()
        .round(1)
    )

    # 4. Data Altair / Vega-Lite / Plotly: 250 sampel representatif bebas missing
    cols_viz = [
        "Order",
        "Neighborhood",
        "Bldg Type",
        "House Style",
        "Overall Qual",
        "Gr Liv Area",
        "Total Bsmt SF",
        "Garage Area",
        "Lot Area",
        "Year Built",
        "SalePrice",
    ]
    viz_clean = df[cols_viz].dropna().copy()
    viz_sample = viz_clean.iloc[::10].head(250).reset_index(drop=True)

    # 5. Data Map: Koordinat deterministik di sekitar Neighborhood Kota Ames, Iowa
    rng = np.random.default_rng(42)
    lat_list = []
    lon_list = []
    for nb in df["Neighborhood"].fillna("NAmes"):
        base_lat, base_lon = AMES_NEIGHBORHOOD_COORDS.get(str(nb), (42.0308, -93.6319))
        lat_list.append(base_lat + rng.normal(0, 0.0028))
        lon_list.append(base_lon + rng.normal(0, 0.0028))

    map_df = pd.DataFrame(
        {
            "lat": lat_list,
            "lon": lon_list,
            "Neighborhood": df["Neighborhood"].values,
            "SalePrice": df["SalePrice"].values,
        }
    )

    return {
        "line_df": line_df,
        "area_df": area_df,
        "bar_df": bar_df,
        "viz_sample": viz_sample,
        "map_df": map_df,
    }


def create_matplotlib_distribution_fig(
    df: pd.DataFrame, custom_color: str = "#1f77b4", save_output: bool = True
) -> plt.Figure:
    """
    Membuat grafik Matplotlib/Seaborn distribusi SalePrice (Mentah vs Log1p)
    dan menyimpannya ke outputs/figures/ sesuai Bagian 9 skill.md.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(df["SalePrice"], kde=True, color=custom_color, ax=axes[0], bins=35)
    axes[0].set_title(
        f"Distribusi SalePrice Mentah (Skew = {df['SalePrice'].skew():.2f})",
        fontweight="bold",
    )
    axes[0].set_xlabel("SalePrice ($)")
    axes[0].set_ylabel("Frekuensi")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    log_sp = np.log1p(df["SalePrice"])
    sns.histplot(log_sp, kde=True, color="#2ca02c", ax=axes[1], bins=35)
    axes[1].set_title(
        f"Distribusi Log1p(SalePrice) (Skew = {log_sp.skew():.2f})",
        fontweight="bold",
    )
    axes[1].set_xlabel("log(1 + SalePrice)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_output:
        fig.savefig(FIGURES_DIR / "distribusi_saleprice.png", dpi=120, bbox_inches="tight")
    return fig


def ensure_media_assets(df: pd.DataFrame) -> Dict[str, Path]:
    """
    Menyiapkan file media lokal (gambar bimbingan.jpg dan file audio sample_audio.wav)
    untuk praktik Bab 2.6 (Menampilkan Media: st.image, st.audio, st.video).
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    img_path = ASSETS_DIR / "bimbingan.jpg"
    if not img_path.exists():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        top_nb = (
            df.groupby("Neighborhood")["SalePrice"]
            .median()
            .sort_values(ascending=False)
            .head(8)
        )
        bars = ax.barh(top_nb.index[::-1], top_nb.values[::-1], color="#2980b9", edgecolor="#1b4f72")
        ax.set_title("Ilustrasi Praktik Bab 2.6: Top 8 Neighborhood (Median SalePrice)", fontweight="bold")
        ax.set_xlabel("Median SalePrice (USD)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 3000, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=8)
        ax.set_xlim(0, top_nb.max() * 1.18)
        plt.tight_layout()
        fig.savefig(img_path, format="jpeg", dpi=120)
        fig.savefig(FIGURES_DIR / "top_neighborhood_median_price.png", dpi=120)
        plt.close(fig)

    wav_path = ASSETS_DIR / "sample_audio.wav"
    if not wav_path.exists():
        sample_rate = 22050
        duration_sec = 2.0
        n_samples = int(sample_rate * duration_sec)
        with wave.open(str(wav_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                freq = 440.0 if t < 1.0 else 554.37  # A4 -> C#5
                envelope = math.exp(-1.5 * (t % 1.0))
                val = int(16000 * envelope * math.sin(2 * math.pi * freq * t))
                frames.extend(struct.pack("<h", val))
            wav_file.writeframes(bytes(frames))

    return {
        "image_path": img_path,
        "audio_path": wav_path,
    }

