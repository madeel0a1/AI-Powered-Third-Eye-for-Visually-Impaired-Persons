"""
AI-Powered "Third Eye" for Visually Impaired Persons - Web App
-----------------------------------------------------------------
Runs the same pipeline as the Kaggle notebook (BLIP scene captioning,
YOLO11 object detection, EasyOCR text extraction, fine-tuned DenseNet121
currency recognition) behind a simple web page: upload an image, get a
combined spoken description back.

Run locally with:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import json
import uuid
from collections import Counter

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from flask import Flask, request, render_template, jsonify, url_for

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
MODEL_DIR = os.path.join(BASE_DIR, "model")

CURRENCY_WEIGHTS_PATH = os.path.join(MODEL_DIR, "currency_densenet121.pth")
CURRENCY_CLASSES_PATH = os.path.join(MODEL_DIR, "currency_classes.json")
CURRENCY_CONFIDENCE_THRESHOLD = 0.75

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------------
# Module 1 - Scene Description (BLIP)
# ---------------------------------------------------------------------------
print("Loading BLIP captioning model... (first run downloads ~1GB, be patient)")
from transformers import BlipProcessor, BlipForConditionalGeneration

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_model.eval()
print("BLIP model loaded.")


def describe_scene(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Module 2 - OCR (EasyOCR)
# ---------------------------------------------------------------------------
print("Loading EasyOCR reader...")
import easyocr

ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
print("EasyOCR reader loaded.")


import numpy as np


def extract_text(image_path: str) -> str:
    # Known EasyOCR issue: passing a file path directly can crash with
    # "too many values to unpack (expected 2)" on some environments because
    # its internal grayscale conversion doesn't always produce a true 2D array.
    # Fix: convert to grayscale ourselves and pass a numpy array instead.
    gray_image = Image.open(image_path).convert("L")
    gray_array = np.array(gray_image)
    results = ocr_reader.readtext(gray_array, detail=0)
    return " ".join(results).strip()


# ---------------------------------------------------------------------------
# Module 2b - Object Detection (YOLO11, pretrained COCO)
# ---------------------------------------------------------------------------
print("Loading YOLO11 object detection model...")
from ultralytics import YOLO

yolo_model = YOLO("yolo11n.pt")
print("YOLO11 model loaded. Classes it can detect:", len(yolo_model.names))


def detect_objects(image_path: str, conf_threshold: float = 0.4) -> str:
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)[0]

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


# ---------------------------------------------------------------------------
# Module 3 - Currency Recognition (fine-tuned DenseNet121, your trained model)
# ---------------------------------------------------------------------------
currency_model = None
currency_classes = []

val_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_currency_model():
    global currency_model, currency_classes

    if not os.path.exists(CURRENCY_WEIGHTS_PATH) or not os.path.exists(CURRENCY_CLASSES_PATH):
        print(
            "WARNING: currency model weights or classes file not found in /model. "
            "Currency recognition will be disabled until you add them. "
            "See README.md for how to export these from your Kaggle notebook."
        )
        return

    with open(CURRENCY_CLASSES_PATH) as f:
        currency_classes = json.load(f)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(currency_classes))
    model.load_state_dict(torch.load(CURRENCY_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    currency_model = model
    print(f"Currency model loaded. Classes: {currency_classes}")


load_currency_model()


def recognize_currency(image_path: str, min_confidence: float = CURRENCY_CONFIDENCE_THRESHOLD) -> str:
    if currency_model is None or not currency_classes:
        return ""

    image = Image.open(image_path).convert("RGB")
    tensor = val_tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = currency_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    if conf.item() < min_confidence:
        return ""

    label = currency_classes[pred_idx.item()]
    return f"{label} currency note detected, confidence {conf.item() * 100:.1f}%"


# ---------------------------------------------------------------------------
# Module 4 - Text-to-Speech (gTTS)
# ---------------------------------------------------------------------------
from gtts import gTTS


def speak(text: str, out_path: str):
    if not text.strip():
        text = "No information detected."
    gTTS(text=text, lang="en").save(out_path)


# ---------------------------------------------------------------------------
# Integrated pipeline
# ---------------------------------------------------------------------------
def third_eye(image_path: str, audio_out_path: str, use_currency: bool = True) -> str:
    scene = describe_scene(image_path)
    objects_found = detect_objects(image_path)
    text_found = extract_text(image_path)
    currency_result = recognize_currency(image_path) if use_currency else ""

    parts = [f"Scene: {scene}."]
    if objects_found:
        parts.append(f"Objects detected: {objects_found}.")
    if text_found:
        parts.append(f"Text detected: {text_found}.")
    if currency_result:
        parts.append(currency_result + ".")

    full_report = " ".join(parts)
    speak(full_report, audio_out_path)
    return full_report


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", currency_available=currency_model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid image file (png/jpg/jpeg/webp/bmp)."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    image_filename = f"{uid}.jpg"
    audio_filename = f"{uid}.mp3"

    image_path = os.path.join(UPLOAD_DIR, image_filename)
    audio_path = os.path.join(AUDIO_DIR, audio_filename)

    try:
        # Normalize every upload to plain RGB JPEG - avoids crashes from RGBA/palette
        # PNGs (transparency, screenshots, etc.) that OCR/YOLO don't expect.
        img = Image.open(file.stream).convert("RGB")
        img.save(image_path, "JPEG", quality=90)
    except Exception as e:
        return jsonify({"error": f"Could not read that image file: {e}"}), 400

    try:
        report = third_eye(image_path, audio_path)
    except Exception as e:
        import traceback
        traceback.print_exc()  # full details in the terminal, for debugging
        return jsonify({"error": f"Processing failed: {e}"}), 500

    return jsonify(
        {
            "report": report,
            "image_url": url_for("static", filename=f"uploads/{image_filename}"),
            "audio_url": url_for("static", filename=f"audio/{audio_filename}"),
        }
    )


if __name__ == "__main__":
    # Note: debug=False (and no reloader) here on purpose - Flask's debug reloader
    # would otherwise re-import this file and reload every ML model twice, roughly
    # doubling startup time. Not needed for running/demoing the app.
    app.run(debug=False, host="0.0.0.0", port=5000)
