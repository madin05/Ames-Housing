import streamlit as st
from PIL import Image

image = Image.open("assets/bimbingan.jpg")
st.image(image, caption="ini adalah gambar")
