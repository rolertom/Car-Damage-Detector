import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("car_damage_model.keras")

st.set_page_config(page_title="Car Damage Detector", layout="wide")

st.markdown("""
<style>
h1, h2 {
    color: black !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d9e4f5 0%, #ffffff 100%);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: #eef3fc !important;
}

.result-box {
    padding: 20px;
    background: #ffffffcc;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>CAR DAMAGE DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload foto mobil untuk mendeteksi kerusakan</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

with col2:
    st.subheader("🔍 Hasil Prediksi")

    if uploaded_file is None:
        st.info("Belum ada gambar yang di-upload.")
    else:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔎 Menganalisis gambar..."):
            pred = model.predict(img_array)[0][0]

        result_html = "<div class='result-box'>"
        
        if pred > 0.5:
            result_html += "<p style='color:green; font-weight:bold;'>✔ Mobil Tidak Rusak</p>"
            confidence = pred
        else:
            result_html += "<p style='color:red; font-weight:bold;'>❗ Kerusakan Terdeteksi</p>"
            confidence = 1 - pred

        result_html += f"<p style='color:black;'>Confidence: <b>{(confidence * 100):.0f}%</b></p>"
        result_html += "</div>"

        st.markdown(result_html, unsafe_allow_html=True)
