import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# ───────────────────────────────────────────────
#  Seite konfigurieren
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Bild-Klassifikator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Bild-Klassifikator mit meinem Keras-Modell")
st.markdown("Lade ein Bild hoch – das Modell sagt dir, was es sieht.")

# ───────────────────────────────────────────────
# Modell und Labels einmalig laden (caching!)
# ───────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    try:
        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        class_names = [name.strip() for name in class_names]  # sauber machen
        return model, class_names
    except Exception as e:
        st.error(f"Modell oder labels.txt konnte nicht geladen werden:\n{e}")
        st.stop()


model, class_names = load_my_model()

# ───────────────────────────────────────────────
# Streamlit Datei-Uploader
# ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Bild hier ablegen oder auswählen …",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # ───────────────────────────────────────────────
    # Preprocessing (genau wie in deinem Originalcode)
    # ───────────────────────────────────────────────
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # in numpy array umwandeln
    image_array = np.asarray(image_resized)

    # Normalisierung (Teachable Machine Standard)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Batch-Dimension hinzufügen → Shape (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ───────────────────────────────────────────────
    # Vorhersage
    # ───────────────────────────────────────────────
    with st.spinner("Modell denkt …"):
        prediction = model.predict(data)
        index = int(np.argmax(prediction))
        class_name = class_names[index]
        confidence = float(prediction[0][index])

    # Klasse bereinigen (entfernt oft "0 ", "1 " usw. am Anfang)
    if class_name[0].isdigit() and class_name[1] == " ":
        class_name = class_name[2:]

    # ───────────────────────────────────────────────
    # Ergebnis schön darstellen
    # ───────────────────────────────────────────────
    st.subheader("Ergebnis")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Klasse:**  {class_name}")
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    # Fortschrittsbalken-Style für die Confidence
    st.progress(confidence)

    # Zusätzliche Info bei niedriger Sicherheit
    if confidence < 0.65:
        st.info("⚠️  Die Vorhersage ist nicht sehr sicher. Vielleicht ein Grenzfall oder das Bild passt nicht gut zu den Trainingsdaten?")

else:
    st.info("Bitte lade ein Bild hoch ↑")

# ───────────────────────────────────────────────
# Kleiner Footer / Hinweis
# ───────────────────────────────────────────────
st.markdown("---")
st.caption("Modell: keras_Model.h5 • Input-Größe: 224×224 • Normalisierung: Teachable Machine Style")
