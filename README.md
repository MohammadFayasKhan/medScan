<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Offline-100%25-00C851?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Tests-66%20Passed-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>

<br/><br/>

```
███╗   ███╗███████╗██████╗ ███████╗ ██████╗ █████╗ ███╗   ██╗     █████╗ ██╗
████╗ ████║██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║    ██╔══██╗██║
██╔████╔██║█████╗  ██║  ██║███████╗██║     ███████║██╔██╗ ██║    ███████║██║
██║╚██╔╝██║██╔══╝  ██║  ██║╚════██║██║     ██╔══██║██║╚██╗██║    ██╔══██║██║
██║ ╚═╝ ██║███████╗██████╔╝███████║╚██████╗██║  ██║██║ ╚████║    ██║  ██║██║
╚═╝     ╚═╝╚══════╝╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═╝  ╚═╝╚═╝
```

# 💊 MedScan AI
### Offline Medicine Intelligence System

*Scan · Search · Chat · Compare — Zero Internet Required*

**[ANTIGRAVITY BUILD]** · INT428 AI Systems Design · Production-Grade Python + Streamlit

---

</div>

## 🎯 What is MedScan AI?

**MedScan AI** is a production-grade, fully-offline AI-powered medicine information system. It combines Computer Vision (OCR), Natural Language Processing, and a multi-strategy search engine to instantly identify and explain medicines — **entirely on your local machine with no internet required at runtime**.

> 🏆 **Key Achievement:** 99.1% Naive Bayes intent classification accuracy across 14 medical intent categories tested with 66 automated unit tests.

---

## ✨ Core Features

| Feature | Technology | Details |
|---------|-----------|---------|
| 📷 **OCR Medicine Scanner** | OpenCV + Tesseract | Adaptive threshold, deskew, noise removal |
| 🔍 **3-Strategy Search Engine** | TF-IDF + Cosine Similarity + Levenshtein | Exact → TF-IDF → Fuzzy fallback |
| 🤖 **Offline NLP Chatbot** | Naive Bayes + TF-IDF | 14 intent categories, 799 training samples |
| ⚖️ **Medicine Comparison** | Radar Chart + Scoring | Compare up to 3 medicines with AI verdict |
| 📋 **Medicine Database** | Pandas + CSV | 20 medicines, 27 data fields each |
| ⬇️ **Export System** | In-memory I/O | CSV, JSON, chat transcript download |
| 🎨 **Premium Dark UI** | Custom CSS + Streamlit | Navy/Cyan/Orange glassmorphism design |

---

## 🖥️ Live Screenshots

<div align="center">

| Home Screen | Search Result |
|:-----------:|:-------------:|
| *Dark themed hero with OCR & text scan modes* | *Exact match card with all 9 info sections* |

</div>

---

## 🏗️ Architecture

```
medscan_ai/
│
├── 🚀 app.py                    # Streamlit entry point (3-tab dashboard)
├── ⚙️  setup.py                  # One-time setup: NLTK + model training
├── 📋 requirements.txt          # All Python dependencies
├── 🐧 run.sh / 🪟 run.bat       # OS-specific launchers
│
├── 🧠 modules/                  # Core AI/NLP Logic Layer
│   ├── preprocessor.py         # NLTK tokenisation + stopword removal + stemming
│   ├── medicine_db.py          # Database loader, validator, query helpers
│   ├── medicine_search.py      # 3-strategy search: Exact, TF-IDF, Fuzzy
│   ├── ocr_engine.py           # OpenCV preprocessing + Tesseract OCR pipeline
│   ├── chatbot.py              # TF-IDF intent classification + response generator
│   ├── intent_classifier.py    # Naive Bayes classifier with joblib persistence
│   ├── compare_engine.py       # Medicine scoring + radar chart + verdict
│   ├── model_trainer.py        # Unified model training orchestrator
│   └── export_utils.py         # CSV / JSON / text transcript export
│
├── 🎨 components/               # Streamlit UI Component Layer
│   ├── ui_styles.py            # Global CSS design system (tokens + animations)
│   ├── medicine_card.py        # Medicine header card with category badge
│   ├── info_sections.py        # 9 collapsible accordion info sections
│   ├── chatbot_ui.py           # MedBot chat interface with quick chips
│   ├── scan_ui.py              # OCR upload + text search interface
│   ├── compare_ui.py           # Comparison tab with table + radar
│   └── sidebar_ui.py           # Sidebar: status, history, quick access
│
├── 📊 data/
│   ├── medicines.csv           # 20 medicines × 27 clinical data fields
│   ├── intents.json            # 14 chatbot intent categories (799 patterns)
│   ├── synonyms.json           # Brand→Generic medicine name mapping
│   └── sample_images/          # OCR test images directory
│
├── 🤖 models/                   # Auto-generated ML model artifacts
│   ├── intent_classifier.pkl   # Trained Naive Bayes intent classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectoriser (chatbot)
│   ├── label_encoder.pkl       # Label encoder for intent classes
│   └── search_index.pkl        # Pre-built TF-IDF medicine search index
│
├── 🧪 tests/                    # IEEE-compliant unit test suite (66 tests)
│   ├── test_db.py              # Database loading, validation, querying
│   ├── test_search.py          # All 3 search strategies + brand resolution
│   ├── test_chatbot.py         # Intent classification + response generation
│   ├── test_ocr.py             # OCR text cleaning + candidate extraction
│   └── test_algorithms.py      # ML accuracy + TF-IDF + compare engine
│
└── 📚 docs/
    └── README.md               # Detailed technical documentation
```

---

## 🤖 ML Algorithms & Pipeline

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│               SEARCH ENGINE                       │
│  ① EXACT MATCH  (name_lower lookup)  → 100%      │
│  ② TFIDF MATCH  (cosine similarity)  → 70-99%    │
│  ③ FUZZY MATCH  (Levenshtein dist.)  → 50-85%    │
└────────────────────┬─────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│            OFFLINE NLP CHATBOT                    │
│  Naive Bayes Classifier → Intent (14 classes)    │
│  TF-IDF Response Templates + Dynamic Fields      │
│  Training Accuracy: 99.1% on 799 samples          │
└────────────────────┬─────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│              OCR PIPELINE                         │
│  Grayscale → CLAHE → Adaptive Threshold →        │
│  Deskew → Morphological Clean → Tesseract LSTM   │
└──────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Tesseract OCR (for image scanning — text search works without it)

### 1 · Clone the repository
```bash
git clone https://github.com/MohammadFayasKhan/medScan.git
cd medScan
```

### 2 · Create & activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate.bat     # Windows
```

### 3 · Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4 · Install Tesseract (for OCR image scanning)
```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows → https://github.com/UB-Mannheim/tesseract/wiki
```

### 5 · Run one-time setup (downloads NLTK, trains models)
```bash
python setup.py
```
> ⚡ Takes ~5 seconds. Outputs model files to `models/`.

### 6 · Launch the app
```bash
streamlit run app.py
```

**App opens at → `http://localhost:8501`**

---

## 🧪 Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Individual modules
python -m pytest tests/test_db.py -v          # Database tests
python -m pytest tests/test_search.py -v      # Search engine tests
python -m pytest tests/test_chatbot.py -v     # NLP chatbot tests
python -m pytest tests/test_ocr.py -v         # OCR pipeline tests
python -m pytest tests/test_algorithms.py -v  # ML algorithm tests

# With coverage report
pip install pytest-cov
python -m pytest tests/ --cov=modules --cov-report=term-missing
```

### Test Results

```
======================== 66 passed in 22.38s ========================
tests/test_algorithms.py   11/11 ✅  NB accuracy: 99.1% (≥85% threshold)
tests/test_chatbot.py      15/15 ✅  All 14 intents classified correctly
tests/test_db.py           16/16 ✅  Load, validate, lookup, filter
tests/test_ocr.py          13/13 ✅  Text cleaning + candidate extraction
tests/test_search.py       11/11 ✅  Exact, TF-IDF, fuzzy, brand name
```

---

## 💊 Medicine Database

20 clinically-detailed medicines across 9 therapeutic categories:

| # | Medicine | Generic Name | Category |
|---|---------|-------------|---------|
| 1 | Paracetamol | Acetaminophen | Analgesic / Antipyretic |
| 2 | Ibuprofen | Ibuprofen | NSAID / Analgesic |
| 3 | Amoxicillin | Amoxicillin trihydrate | Beta-Lactam Antibiotic |
| 4 | Metformin | Metformin HCl | Biguanide / Antidiabetic |
| 5 | Hypromellose | HPMC | Ophthalmic Lubricant |
| 6 | Cetirizine | Cetirizine HCl | H1 Antihistamine |
| 7 | Omeprazole | Omeprazole | Proton Pump Inhibitor |
| 8 | Azithromycin | Azithromycin dihydrate | Macrolide Antibiotic |
| 9 | Amlodipine | Amlodipine besylate | Calcium Channel Blocker |
| 10 | Atorvastatin | Atorvastatin calcium | HMG-CoA Reductase Inhibitor |
| 11 | Pantoprazole | Pantoprazole sodium | Proton Pump Inhibitor |
| 12 | Dextromethorphan | DXM hydrobromide | Antitussive |
| 13 | Losartan | Losartan potassium | ARB / Antihypertensive |
| 14 | Aspirin | Acetylsalicylic acid | Antiplatelet / NSAID |
| 15 | Montelukast | Montelukast sodium | Leukotriene Antagonist |
| 16 | Salbutamol | Salbutamol sulphate | SABA / Bronchodilator |
| 17 | Diclofenac | Diclofenac sodium | NSAID / Anti-inflammatory |
| 18 | Clonazepam | Clonazepam | Benzodiazepine / Antiepileptic |
| 19 | Aceclofenac | Aceclofenac | NSAID / Analgesic |
| 20 | Cholecalciferol | Vitamin D3 | Vitamin / Bone Health |

Each medicine includes **27 clinical data fields**: uses, mechanism, dosage, timing, warnings, contraindications, interactions, side effects, substitutes, pack sizes, and sources.

---

## 📋 9-Section Medicine Information Layout

When a medicine is found, MedScan AI displays:

```
ℹ️  Basic Information      ←  Name, generic, form, strength, manufacturer
⚡  Usage & Action         ←  Uses, mechanism, medical indications
💧  Dosage & Use           ←  Dose, timing, administration tips, spacing
⚠️  Warnings               ←  Pregnancy, paediatric, driving, storage
🚫  Contraindications      ←  Conditions where medicine must NOT be used
🔗  Drug Interactions      ←  Known medicine-medicine interactions
🔴  Side Effects           ←  Common (amber) and serious (red) effects
🛍️  Availability            ←  Pack sizes and therapeutic substitutes
📚  Sources                ←  Reference databases used
```

---

## 🔒 Offline-First Guarantee

| Check | Status |
|-------|--------|
| API Keys Required | ❌ None |
| Internet at Runtime | ❌ Not Used |
| ML Training | ✅ Local (`setup.py`) |
| NLTK Resources | ✅ Cached locally |
| Tesseract OCR | ✅ On-device |
| All Data Storage | ✅ `data/` + `models/` |

---

## 🛠️ Technology Stack

```
Language     →  Python 3.9+
Frontend     →  Streamlit 1.50 + Custom CSS (dark navy/cyan/orange)
ML / NLP     →  scikit-learn 1.6 (TF-IDF, Naive Bayes, Cosine Similarity)
NLP Toolkit  →  NLTK 3.9 (tokenisation, stopwords, stemming)
OCR          →  OpenCV 4.13 + Tesseract 5.x (LSTM engine)
Fuzzy Match  →  fuzzywuzzy + python-Levenshtein
Data Layer   →  Pandas 2.3 + CSV
Charts       →  Matplotlib 3.9 (radar chart)
ML Persist   →  joblib (model serialisation)
Testing      →  pytest 8.4 (66 unit tests)
```

---

## 📁 Project Hygiene

- ✅ All files follow PEP 8 with IEEE-style docstrings
- ✅ No hardcoded paths — all paths are resolved relative to `PROJECT_ROOT`
- ✅ `.gitignore` excludes `venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`
- ✅ `models/` excluded from git (auto-generated by `setup.py`)
- ✅ Zero external API dependencies

---

## ⚠️ Medical Disclaimer

> This application is for **educational and informational purposes only**.
> It is **not a substitute for professional medical advice, diagnosis, or treatment**.
> Always consult a qualified healthcare professional before making any medical decision.
> In emergencies, call **108** (India) or your local emergency number.

---

## 👨‍💻 Author

**Mohammad Fayas Khan**
- 📧 [MohammadFayasKhan](https://github.com/MohammadFayasKhan)
- 🏫 INT428 — AI Systems Design
- 🏗️ ANTIGRAVITY BUILD System

---

<div align="center">

**⭐ Star this repo if you found it useful!**

*Built with 💊 and Python — 100% Offline · Zero API Keys · Production Grade*

</div>
