# MEDSCAN AI

> **100% Offline Medicine Intelligence System**
> ANTIGRAVITY BUILD | Python + Streamlit | Academic IEEE Standard

---

## 🎯 What is MEDSCAN AI?

MEDSCAN AI is a production-grade, fully offline medicine information system that allows users to:

- 📷 **Scan** medicine packages using OCR (OpenCV + Tesseract)
- 🔍 **Search** medicines by name, brand name, or generic name
- 💬 **Chat** with an AI bot (MedBot) that answers clinical questions offline
- ⚖️ **Compare** up to 3 medicines in a side-by-side analysis
- 📋 **Browse** all 20 medicines in the local database
- ⬇️ **Export** medicine data as CSV, JSON, or chat transcripts

**Zero internet required at runtime. All ML runs locally.**

---

## 🖥️ System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10+ / macOS 12+ / Ubuntu 20.04+ |
| Python | 3.9 or higher |
| RAM | Minimum 4GB (8GB recommended) |
| Disk | ~500MB (models + dependencies) |
| Tesseract | Required for OCR (text mode works without it) |

---

## ⚡ Quick Start

### Step 1: Clone / Download
```bash
cd Int428Project/medscan_ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate.bat     # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract (for OCR image scanning)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows — download from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 5: Run Setup (first time only)
```bash
python setup.py
```

This downloads NLTK resources, trains the Naive Bayes classifier, and builds the search index.  
Takes approximately 15–30 seconds.

### Step 6: Launch the App
```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 🧠 Architecture Overview

```
app.py                         ← Streamlit entry point
├── components/
│   ├── ui_styles.py           ← All custom CSS (design tokens + animations)
│   ├── medicine_card.py       ← Medicine header card renderer
│   ├── info_sections.py       ← 9 collapsible info sections
│   ├── chatbot_ui.py          ← MedBot chat interface
│   ├── scan_ui.py             ← Scan/search interface
│   ├── compare_ui.py          ← Comparison tab
│   └── sidebar_ui.py          ← Sidebar renderer
└── modules/
    ├── preprocessor.py        ← NLTK text preprocessing pipeline
    ├── medicine_db.py         ← CSV database loading + queries
    ├── medicine_search.py     ← 3-strategy search engine
    ├── ocr_engine.py          ← OpenCV + Tesseract OCR pipeline
    ├── chatbot.py             ← TF-IDF intent classification + responses
    ├── intent_classifier.py   ← Naive Bayes intent classifier
    ├── compare_engine.py      ← Scoring + radar chart + verdict
    ├── model_trainer.py       ← Unified model training orchestrator
    └── export_utils.py        ← CSV/JSON/text export
```

---

## 🤖 ML Algorithms Used

| Component | Algorithm | Library |
|-----------|-----------|---------|
| Medicine Search | TF-IDF + Cosine Similarity | scikit-learn |
| Intent Classification | Naive Bayes Classifier | scikit-learn |
| Fuzzy Matching (OCR errors) | Levenshtein Distance | fuzzywuzzy |
| Text Preprocessing | Tokenisation + Stopword Removal + Stemming | NLTK |
| OCR Pipeline | Adaptive Threshold + Hough Deskew | OpenCV |
| OCR Engine | LSTM Neural Network | Tesseract |

---

## 💊 Medicine Database

20 medicines across 9 categories with 27 data fields each:

| Category | Medicines |
|----------|-----------|
| Analgesic / Antipyretic | Paracetamol, Aspirin |
| NSAID | Ibuprofen, Diclofenac, Aceclofenac |
| Antibiotic | Amoxicillin, Azithromycin |
| Antidiabetic | Metformin |
| Ophthalmic | Hypromellose (Eye Drops) |
| Antihistamine | Cetirizine |
| PPI | Omeprazole, Pantoprazole |
| Antihypertensive | Amlodipine, Losartan |
| Statin | Atorvastatin |
| Bronchodilator / LABA | Salbutamol |
| Vitamin | Cholecalciferol (Vit D3) |
| Antiepileptic | Clonazepam |
| Antitussive | Dextromethorphan |
| Leukotriene Antagonist | Montelukast |

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_db.py -v
pytest tests/test_search.py -v
pytest tests/test_chatbot.py -v
pytest tests/test_ocr.py -v
pytest tests/test_algorithms.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=modules --cov-report=term-missing
```

---

## 🔒 Offline-First Guarantee

- ✅ All ML models trained locally via `setup.py`
- ✅ No API keys required
- ✅ No `requests` calls made at runtime
- ✅ NLTK resources cached locally after first download
- ✅ All data stored in `data/` and `models/` directories
- ✅ Tesseract OCR runs locally on device

---

## ⚠️ Medical Disclaimer

This application is for **educational and informational purposes only**.
It is **not a substitute for professional medical advice**.
Always consult a qualified healthcare professional before making
any medical decisions. In emergencies, dial **108** (India) or 
your local emergency number.

---

## 📂 Project Structure

```
medscan_ai/
├── app.py                # Main Streamlit entry point
├── setup.py              # One-time setup/training script
├── requirements.txt      # Python dependencies
├── run.sh                # Linux/Mac launcher
├── run.bat               # Windows launcher
│
├── .streamlit/
│   └── config.toml       # Dark theme Streamlit config
│
├── data/
│   ├── medicines.csv     # 20-medicine offline database
│   ├── intents.json      # 14 NLP chatbot intent categories
│   ├── synonyms.json     # Brand→generic name mapping
│   └── sample_images/    # Test images for OCR
│
├── models/               # Trained ML models (auto-generated)
│   ├── intent_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── search_index.pkl
│
├── modules/              # Core logic modules
│   ├── preprocessor.py
│   ├── medicine_db.py
│   ├── medicine_search.py
│   ├── ocr_engine.py
│   ├── chatbot.py
│   ├── intent_classifier.py
│   ├── compare_engine.py
│   ├── model_trainer.py
│   └── export_utils.py
│
├── components/           # Streamlit UI components
│   ├── ui_styles.py
│   ├── medicine_card.py
│   ├── info_sections.py
│   ├── chatbot_ui.py
│   ├── scan_ui.py
│   ├── compare_ui.py
│   └── sidebar_ui.py
│
├── docs/                 # Documentation
│   ├── README.md         # This file
│   ├── DATASET_GUIDE.md  # How to expand the database
│   └── ARCHITECTURE.md   # Full system design document
│
└── tests/                # Unit tests
    ├── test_db.py
    ├── test_search.py
    ├── test_chatbot.py
    ├── test_ocr.py
    └── test_algorithms.py
```

---

## 🎓 Academic Information

**Course:** INT428 — AI Systems Design  
**Build System:** ANTIGRAVITY BUILD  
**Version:** 1.0.0  
**Coding Standard:** IEEE 1471 compliant  
**Documentation:** Full inline docstrings, IEEE-style module headers
