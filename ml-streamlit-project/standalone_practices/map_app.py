import streamlit as st
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
df = pd.DataFrame(
    rng.standard_normal((500, 2)) / [150, 150] + [42.0308, -93.6319],
    columns=["lat", "lon"],
)
st.map(df)
