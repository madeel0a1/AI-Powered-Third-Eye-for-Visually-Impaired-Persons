# 👁️ Third Eye — AI-Powered Assistant for Visually Impaired Persons

**An end-to-end computer vision pipeline that turns a single photo into a spoken description of the world — scene, objects and their position, readable text, and currency denomination — built and deployed as a live web app.**

**Author:** [Muhammad Adeel](https://github.com/madeel0a1) · [LinkedIn](https://www.linkedin.com/in/muhammad-adeel-ml)

---



## 💡 The Problem

Visually impaired individuals often depend on others to describe their surroundings, read text on packaging or signage, or verify currency notes when handling cash — everyday tasks that limit independence. Third Eye combines four separate AI capabilities into a single assistive pipeline that narrates all of this out loud from one photo.

## ⚙️ How It Works

| Module | Model | What it does |
|---|---|---|
| 🖼️ **Scene Description** | BLIP (Salesforce, CNN + Transformer) | Generates a natural-language caption of the overall scene |
| 📦 **Object Detection** | YOLO11 (pretrained on COCO, 80 classes) | Detects specific objects and their rough position (left / center / right) |
| 📝 **Text Extraction** | EasyOCR | Reads any visible printed text — signs, labels, packaging |
| 💵 **Currency Recognition** | DenseNet121 (fine-tuned, custom-trained) | Identifies Pakistani Rupee notes — 10 to 5000, front & back |
| 🔊 **Text-to-Speech** | gTTS | Converts the combined report into spoken audio |

All four outputs are merged into a single report and read aloud — so the result is useful whether the user reads it or hears it.

## 🛠️ Tech Stack

**ML / CV:** PyTorch · torchvision · Transformers (BLIP) · Ultralytics (YOLO11) · EasyOCR
**App:** Streamlit
**Audio:** gTTS
**Training:** Transfer learning (DenseNet121, ImageNet weights) on Kaggle with GPU

## 📊 Currency Model — Training Details

The DenseNet121 classifier was fine-tuned from ImageNet weights on a labeled dataset of Pakistani currency notes across **14 classes** (10 / 20 / 50 / 100 / 500 / 1000 / 5000 Rupees — front and back of each). Data augmentation (random horizontal flip, ±10° rotation) was used to improve generalization, reaching **~99.6% training accuracy** over 5 epochs.

Predictions below a **75% confidence threshold** are treated as "not a currency note" rather than forced into the closest class — this avoids confidently mislabeling unrelated images, a deliberate design choice for an assistive tool where wrong answers are worse than no answer.

## 📁 Project Structure

```
├── streamlit_app.py            # Streamlit app - full pipeline + UI
├── app.py                      # Flask version (alternative, for local/Render deployment)
├── requirements.txt
├── packages.txt                 # System-level deps for Streamlit Cloud
├── .streamlit/config.toml       # App theme
├── templates/index.html         # Frontend for the Flask version
└── model/
    ├── currency_densenet121.pth # Trained currency classifier weights
    └── currency_classes.json    # The 14 class labels
```

## 🖥️ Run Locally

**Requirements:** Python 3.9–3.12

```bash
git clone https://github.com/madeel0a1/AI-Powered-Third-Eye-for-Visually-Impaired-Persons.git
cd AI-Powered-Third-Eye-for-Visually-Impaired-Persons

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

> First run downloads the pretrained BLIP and YOLO11 weights automatically (~1–2 min, one-time only, then cached).

## ☁️ Deploy Your Own Copy (Streamlit Community Cloud — free)

1. Fork/push this repo to your own GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. **New app** → select this repo → set main file path to `streamlit_app.py` → **Deploy**.
4. First deploy takes a few minutes (installing PyTorch, downloading pretrained weights on first request). Done — you get a public `*.streamlit.app` link.

## 🔭 Possible Future Improvements

- Live webcam mode instead of single-image upload
- Urdu text-to-speech option
- Larger currency dataset for improved validation accuracy
- Swap BLIP-base for BLIP-large for richer scene descriptions
- Object detection fine-tuned on local/Pakistani-specific objects beyond COCO's 80 classes

---

*Built as part of my transition into applied AI / computer vision, combining transfer learning, model fine-tuning, and full-stack deployment in one project.*
