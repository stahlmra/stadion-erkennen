import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Seitentitel
st.title("🏟️ Stadion Erkennung App")
st.write("Lade ein Bild hoch und das Modell erkennt das Stadion!")

# Modell laden
@st.cache_resource
def load_my_model():
    return load_model("keras_Model.h5", compile=False)

model = load_my_model()

# Labels laden
class_names = open("labels.txt", "r").readlines()

# Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Stadionbild...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader("🔍 Ergebnis:")
    st.write(f"**Stadion:** {class_name}")
    st.write(f"**Sicherheit:** {confidence_score * 100:.2f}%")
