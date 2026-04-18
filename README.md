<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Offline-100%25-00C851?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Tests-66%20Passed-brightgreen?style=for-the-badge"/>

<br/><br/>

# 💊 MedScan AI

### Offline Medicine Intelligence System

*Scan · Search · Chat · Compare — Zero Internet Required at Runtime*

**INT428 — AI Systems Design**

---

</div>

## 🧭 Overview

**MedScan AI** is a fully offline, AI-powered medicine information system built with Python and Streamlit. It lets users identify medicines by uploading a photo or typing a name, then displays detailed clinical information and allows follow-up questions through a locally running NLP chatbot — all without any internet connection or external API.

The project was built as part of the **INT428 AI Systems Design** course and demonstrates real-world application of machine learning concepts including TF-IDF vectorisation, Naive Bayes classification, computer vision, and fuzzy text matching.

> ✅ **Highlight:** 99.1% Naive Bayes intent classification accuracy across 14 medical categories, tested with 66 automated unit tests.

---

## ✨ Features

| Feature | Technology Used |
|---------|----------------|
| 📷 **OCR Medicine Scanner** | OpenCV image preprocessing + Tesseract LSTM |
| 🔍 **3-Strategy Search Engine** | Exact match → TF-IDF cosine similarity → Levenshtein fuzzy |
| 🤖 **Offline NLP Chatbot** | Naive Bayes classifier + TF-IDF intent indexing |
| ⚖️ **Side-by-Side Comparison** | Radar chart scoring + written verdict |
| 📋 **Medicine Database Browser** | 20 medicines × 27 clinical fields, filterable by category |
| ⬇️ **Data Export** | Download results as CSV, JSON, or chat transcript |
| 🎨 **Custom Dark UI** | Tailored CSS design system, no external CSS framework |

---

## 📁 Project Structure

```
medscan_ai/
│
├── app.py                       # Main entry point — Streamlit 3-tab dashboard
├── setup.py                     # First-time setup: downloads NLTK, trains models
├── requirements.txt             # All Python package dependencies
├── packages.txt                 # System packages for Hugging Face Spaces (apt-get)
├── Dockerfile                   # Builds a fully self-contained Docker image
├── docker-compose.yml           # One-command Docker launch with volume persistence
├── Makefile                     # Shortcuts: make run, make stop, make test, etc.
├── run.sh / run.bat             # Launch shortcuts for Linux/Mac and Windows
│
├── modules/                     # Core logic — all AI/NLP processing lives here
│   ├── preprocessor.py         # Text cleaning, tokenisation, stopword removal (NLTK)
│   ├── medicine_db.py          # Database loader, validator, query helpers
│   ├── medicine_search.py      # Multi-strategy search: exact, TF-IDF, fuzzy
│   ├── ocr_engine.py           # OpenCV image preprocessing + Tesseract OCR
│   ├── chatbot.py              # TF-IDF intent matcher + response generator
│   ├── intent_classifier.py    # Naive Bayes intent classifier (sklearn)
│   ├── compare_engine.py       # Scoring logic + matplotlib radar chart
│   ├── model_trainer.py        # Orchestrates model training at setup time
│   └── export_utils.py         # In-memory CSV/JSON/text export for Streamlit
│
├── components/                  # UI layer — all Streamlit rendering code
│   ├── ui_styles.py            # Global CSS injected into Streamlit
│   ├── medicine_card.py        # Medicine header card with category badge
│   ├── info_sections.py        # 9 collapsible accordion info sections
│   ├── chatbot_ui.py           # Chat interface with quick question chips
│   ├── scan_ui.py              # Upload/text search interface and OCR results
│   ├── compare_ui.py           # Comparison tab with table and radar chart
│   └── sidebar_ui.py           # Sidebar: system status, history, quick access
│
├── data/
│   ├── medicines.csv           # 20 medicines with 27 clinical data fields each
│   ├── intents.json            # 14 chatbot intent categories, 799 patterns total
│   ├── synonyms.json           # Brand name to generic name mapping
│   └── sample_images/          # Folder for test images when using OCR mode
│
├── models/                      # Auto-generated at setup — not committed to Git
│   ├── intent_classifier.pkl   # Trained Naive Bayes classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectoriser (chatbot)
│   ├── label_encoder.pkl       # Intent label encoder
│   └── search_index.pkl        # Pre-built medicine search index
│
├── tests/                       # Unit test suite — 66 tests across 5 modules
│   ├── test_db.py              # Tests for database loading and querying
│   ├── test_search.py          # Tests for all 3 search strategies
│   ├── test_chatbot.py         # Tests for intent classification and responses
│   ├── test_ocr.py             # Tests for OCR text cleaning and candidates
│   └── test_algorithms.py      # Tests for ML accuracy and algorithm correctness
│
└── docs/
    └── README.md               # Extended technical documentation
```

---

## 🤖 How the AI Works

### Search Engine — 3 Strategies

When the user types a medicine name, the search runs three strategies in order:

```
User Input
    │
    ▼
① EXACT MATCH        → Checks if input matches any name_lower in database (O(1))
    │ not found
    ▼
② TF-IDF SEARCH      → Vectorises query, computes cosine similarity against corpus
    │ low confidence
    ▼
③ FUZZY MATCH        → Levenshtein distance on all names (handles typos, OCR errors)
    │
    ▼
Result + Confidence Score + Search Strategy Label
```

The name is also checked against a synonyms dictionary so brand names like "Crocin" resolve to "Paracetamol".

### Chatbot — Naive Bayes + TF-IDF

The chatbot classifies the user's question into one of 14 medical intents:

```
User Message: "Is it safe to take this during pregnancy?"
    │
    ▼
Preprocessing: lowercase → tokenise → remove stopwords
    │
    ▼
TF-IDF Vectorise → Cosine Similarity against 799 training patterns
    │
    ▼
Naive Bayes Classifier → Intent: "pregnancy" (confidence: 0.93)
    │
    ▼
Response Generator → Fills template with medicine-specific data fields
    │
    ▼
Response: "Paracetamol is considered safe in all trimesters..."
```

### OCR Pipeline

```
Uploaded Image
    │
    ▼
Grayscale → CLAHE contrast enhancement → Adaptive threshold
    │
    ▼
Deskew (Hough line detection) → Morphological noise removal
    │
    ▼
Tesseract LSTM OCR → Raw text
    │
    ▼
OCR error correction (0→O, 1→I) → Candidate extraction → Database search
```

---

## ⚡ Quick Start

### Requirements
- Python 3.9 or higher
- Tesseract OCR (only needed for image scanning — text search works without it)

### Step 1: Clone
```bash
git clone https://github.com/MohammadFayasKhan/medScan.git
cd medScan
```

### Step 2: Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate.bat     # Windows
```

### Step 3: Install Python packages
```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract (for OCR)
```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows — installer at:
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 5: Run setup (first time only — ~5 seconds)
```bash
python setup.py
```

This downloads NLTK data, trains the Naive Bayes classifier, and builds the TF-IDF search index. All artifacts are saved to `models/`.

### Step 6: Launch
```bash
streamlit run app.py
```

App opens at **`http://localhost:8501`**

---

## 🧪 Running Tests

```bash
# Run all 66 tests
python -m pytest tests/ -v

# Run individual test files
python -m pytest tests/test_db.py -v
python -m pytest tests/test_search.py -v
python -m pytest tests/test_chatbot.py -v
python -m pytest tests/test_ocr.py -v
python -m pytest tests/test_algorithms.py -v

# With coverage report
pip install pytest-cov
python -m pytest tests/ --cov=modules --cov-report=term-missing
```

### Test Results

```
======================== 66 passed in 22.38s ========================

Module                  Tests   Result   Notes
──────────────────────  ─────   ──────   ──────────────────────────
test_algorithms.py       11     ✅ Pass  NB accuracy: 99.1%
test_chatbot.py          15     ✅ Pass  All 14 intents matched
test_db.py               16     ✅ Pass  Load, validate, query, filter
test_ocr.py              13     ✅ Pass  Text clean + candidate extract
test_search.py           11     ✅ Pass  Exact, TF-IDF, fuzzy, brand
```

---

## 💊 Medicine Database

20 medicines are included, each with 27 clinical data fields:

| Medicine | Category |
|---------|---------|
| Paracetamol | Analgesic / Antipyretic |
| Ibuprofen | NSAID / Analgesic |
| Amoxicillin | Beta-Lactam Antibiotic |
| Metformin | Biguanide / Antidiabetic |
| Hypromellose | Ophthalmic Lubricant |
| Cetirizine | H1 Antihistamine |
| Omeprazole | Proton Pump Inhibitor |
| Azithromycin | Macrolide Antibiotic |
| Amlodipine | Calcium Channel Blocker |
| Atorvastatin | HMG-CoA Reductase Inhibitor |
| Pantoprazole | Proton Pump Inhibitor |
| Dextromethorphan | Antitussive |
| Losartan | ARB / Antihypertensive |
| Aspirin | Antiplatelet / NSAID |
| Montelukast | Leukotriene Antagonist |
| Salbutamol | SABA / Bronchodilator |
| Diclofenac | NSAID / Anti-inflammatory |
| Clonazepam | Benzodiazepine / Antiepileptic |
| Aceclofenac | NSAID / Analgesic |
| Cholecalciferol | Vitamin D3 / Bone Health |

Each medicine includes: generic name, brand names, form, strength, mechanism, uses, indications, dosage, timing, administration tips, pregnancy warning, paediatric warning, driving warning, storage, contraindications, drug interactions, common & serious side effects, substitutes, pack sizes, and sources.

---

## 🔒 Offline Guarantee

| Check | Status |
|-------|--------|
| API Keys Required | ❌ None |
| Internet at Runtime | ❌ Not Used |
| ML Model Training | ✅ Local (setup.py) |
| NLTK Resources | ✅ Downloaded once, cached locally |
| OCR Engine | ✅ Tesseract runs on-device |
| All Data Files | ✅ Stored in `data/` and `models/` |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Web Framework | Streamlit 1.50 |
| Machine Learning | scikit-learn 1.6 (TF-IDF, Naive Bayes) |
| NLP Toolkit | NLTK 3.9 |
| Computer Vision | OpenCV 4.13 |
| OCR Engine | Tesseract 5.x |
| Fuzzy Matching | fuzzywuzzy + python-Levenshtein |
| Data Layer | Pandas 2.3 |
| Visualisation | Matplotlib 3.9 |
| Model Persistence | joblib |
| Testing | pytest 8.4 |
| Containerisation | Docker + Docker Compose |

---

## 🐳 Docker Deployment (Fully Offline, No Python Needed)

Docker packages the entire app — Python, all libraries, Tesseract OCR, NLTK data, and trained ML models — into a single container image. Anyone with Docker installed can run it **without installing Python or any dependencies**.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows)
- Or Docker Engine (Linux)

### Option A: One command with Docker Compose (recommended)

```bash
git clone https://github.com/MohammadFayasKhan/medScan.git
cd medScan

# Build the image and start the container
docker-compose up --build
```

App opens at → **`http://localhost:8501`**

To stop:
```bash
docker-compose down
```

### Option B: Manual Docker commands

```bash
# Build the image (takes ~3-5 minutes on first build)
docker build -t medscan-ai .

# Run the container
docker run -p 8501:8501 medscan-ai
```

### Option C: Makefile shortcuts

```bash
make run     # Build and start with docker-compose
make stop    # Stop the container
make logs    # Stream container logs
make shell   # Open bash inside the container
make clean   # Remove image, container, and volumes
make test    # Run the test suite locally
```

### What happens inside the Docker build

```
Step 1: Start from python:3.9-slim base image
Step 2: apt-get install tesseract-ocr, libgl1-mesa-glx, etc.
Step 3: pip install -r requirements.txt
Step 4: COPY all project files
Step 5: python setup.py → downloads NLTK, trains Naive Bayes, builds TF-IDF index
Step 6: EXPOSE 8501
Step 7: streamlit run app.py
```

Once the image is built, **it never needs the internet again**.

---

## 🌐 Hugging Face Spaces (Public Online Demo)

Hugging Face Spaces hosts the app at a public URL so anyone can try it **from any browser without installing anything**.

### How to deploy your own Space

**Step 1:** Create a free account at [huggingface.co](https://huggingface.co)

**Step 2:** Go to [huggingface.co/new-space](https://huggingface.co/new-space) and fill in:
- **Space name:** `medscan-ai`  
- **SDK:** `Streamlit`  
- **Visibility:** Public

**Step 3:** Push this repository to your Space:

```bash
# Add HF Space as a second remote
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/medscan-ai

# Push to Hugging Face
git push hf main
```

**Step 4:** Hugging Face automatically:
1. Reads `packages.txt` → installs `tesseract-ocr` via apt
2. Reads `requirements.txt` → installs all Python packages
3. Runs `streamlit run app.py`
4. On first boot, `app.py` detects missing models and runs `setup.py` automatically

Your app is live at: `https://huggingface.co/spaces/YOUR_HF_USERNAME/medscan-ai`

### Key files for Hugging Face Spaces

| File | Purpose |
|------|---------|
| `packages.txt` | System packages installed via `apt-get` (Tesseract, libgl1) |
| `requirements.txt` | Python packages installed via `pip` |
| `app.py` | Entry point — HF Spaces runs `streamlit run app.py` |
| `.streamlit/config.toml` | Theme and server settings |

> **Note:** HF Spaces has internet access during the build phase, so NLTK downloads and model training work automatically. The app itself runs offline once deployed.

---

## 🔒 Offline Guarantee

| Check | Status |
|-------|--------|
| API Keys Required | ❌ None |
| Internet at Runtime | ❌ Not Used |
| ML Model Training | ✅ Local (setup.py or Docker build) |
| NLTK Resources | ✅ Downloaded once, cached locally |
| OCR Engine | ✅ Tesseract runs on-device |
| All Data Files | ✅ Stored in `data/` and `models/` |
| Docker Image | ✅ Fully self-contained after build |

---

## ⚠️ Medical Disclaimer

> This application is for **educational and academic purposes only**. The information provided is **not a substitute for professional medical advice**, diagnosis, or treatment. Always consult a qualified healthcare professional before taking, stopping, or changing any medication. In medical emergencies, call **108** (India) or your local emergency number.

---

## 👨‍💻 Author

**Mohammad Fayas Khan**
- GitHub: [@MohammadFayasKhan](https://github.com/MohammadFayasKhan)
- Course: INT428 — AI Systems Design

---

<div align="center">

*Built with Python · Runs 100% Offline · Docker Ready · HF Spaces Compatible*

</div>

