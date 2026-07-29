"""
AI-Powered "Third Eye" for Visually Impaired Persons - Streamlit App
-----------------------------------------------------------------------
Same pipeline as the Kaggle notebook (BLIP scene captioning, YOLO11 object
detection, EasyOCR text extraction, fine-tuned DenseNet121 currency
recognition) — packaged as a Streamlit app for one-click deployment on
Streamlit Community Cloud.

Run locally with:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import os
import json
import tempfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
from gtts import gTTS

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Third Eye — AI Assistant for the Visually Impaired",
    page_icon="👁️",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CURRENCY_WEIGHTS_PATH = os.path.join(MODEL_DIR, "currency_densenet121.pth")
CURRENCY_CLASSES_PATH = os.path.join(MODEL_DIR, "currency_classes.json")
CURRENCY_CONFIDENCE_THRESHOLD = 0.75

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model loading (cached so this only runs once per session, not per upload)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading scene description model...")
def load_blip():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading text reader (OCR)...")
def load_ocr():
    import easyocr

    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


@st.cache_resource(show_spinner="Loading object detection model...")
def load_yolo():
    from ultralytics import YOLO

    return YOLO("yolo11n.pt")


@st.cache_resource(show_spinner="Loading currency recognition model...")
def load_currency_model():
    if not os.path.exists(CURRENCY_WEIGHTS_PATH) or not os.path.exists(CURRENCY_CLASSES_PATH):
        return None, []

    with open(CURRENCY_CLASSES_PATH) as f:
        classes = json.load(f)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.load_state_dict(torch.load(CURRENCY_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, classes


val_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------
def describe_scene(image: Image.Image, processor, model) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)


def extract_text(image: Image.Image, reader) -> str:
    # Convert to grayscale numpy array ourselves - passing a PIL image or file
    # path directly can crash EasyOCR on some environments with
    # "too many values to unpack (expected 2)".
    gray_array = np.array(image.convert("L"))
    results = reader.readtext(gray_array, detail=0)
    return " ".join(results).strip()


def detect_objects(image: Image.Image, yolo_model, conf_threshold: float = 0.4) -> str:
    results = yolo_model(image, conf=conf_threshold, verbose=False)[0]

    if len(results.boxes) == 0:
        return ""

    img_width = results.orig_shape[1]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        x_center = float(box.xywh[0][0])

        if x_center < img_width / 3:
            position = "on your left"
        elif x_center > 2 * img_width / 3:
            position = "on your right"
        else:
            position = "ahead"

        detections.append((label, position))

    counts = Counter(detections)
    parts = []
    for (label, position), count in counts.items():
        if count > 1:
            parts.append(f"{count} {label}s {position}")
        else:
            parts.append(f"a {label} {position}")

    return ", ".join(parts)


def recognize_currency(image: Image.Image, model, classes, min_confidence: float = CURRENCY_CONFIDENCE_THRESHOLD) -> str:
    if model is None or not classes:
        return ""

    tensor = val_tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    if conf.item() < min_confidence:
        return ""

    label = classes[pred_idx.item()]
    return f"{label} currency note detected, confidence {conf.item() * 100:.1f}%"


def speak(text: str) -> str:
    if not text.strip():
        text = "No information detected."
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    gTTS(text=text, lang="en").save(tmp_path)
    return tmp_path


def third_eye(image: Image.Image, blip_processor, blip_model, ocr_reader, yolo_model, currency_model, currency_classes):
    scene = describe_scene(image, blip_processor, blip_model)
    objects_found = detect_objects(image, yolo_model)
    text_found = extract_text(image, ocr_reader)
    currency_result = recognize_currency(image, currency_model, currency_classes)

    parts = [f"Scene: {scene}."]
    if objects_found:
        parts.append(f"Objects detected: {objects_found}.")
    if text_found:
        parts.append(f"Text detected: {text_found}.")
    if currency_result:
        parts.append(currency_result + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("👁️ Third Eye")
st.caption("AI assistant for visually impaired persons — upload a photo and hear what's in it.")

blip_processor, blip_model = load_blip()
ocr_reader = load_ocr()
yolo_model = load_yolo()
currency_model, currency_classes = load_currency_model()

if currency_model is None:
    st.warning(
        "Currency recognition model files weren't found in the `model/` folder — "
        "scene description, object detection, and text reading still work normally."
    )

uploaded_file = st.file_uploader(
    "Upload a photo", type=["png", "jpg", "jpeg", "webp", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        report = third_eye(
            image, blip_processor, blip_model, ocr_reader, yolo_model,
            currency_model, currency_classes,
        )
        audio_path = speak(report)

    st.subheader("Third Eye report")
    st.write(report)

    with open(audio_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

    os.remove(audio_path)
else:
    st.info("Choose an image above to get started.")

st.markdown("---")
st.caption(
    "Built by Muhammad Adeel — Scene captioning (BLIP), object detection (YOLO11), "
    "OCR (EasyOCR), currency recognition (fine-tuned DenseNet121)."
)
