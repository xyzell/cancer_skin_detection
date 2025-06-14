import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

st.title("Deteksi Kanker Kulit dengan CNN")
st.write("Upload gambar kulit dan sistem akan memprediksi apakah itu kanker atau bukan.")

uploaded_file = st.file_uploader("Upload gambar kulit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img_array = np.array(image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch)[0][0]
    label = "Kanker Kulit" if prediction > 0.5 else "Bukan Kanker Kulit"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediksi: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}")
