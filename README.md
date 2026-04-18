<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Offline-100%25-00C851?style=for-the-badge"/>

<br/><br/>

# 💊 MedScan+ 

### Production-Grade Offline Medicine Intelligence System

*Scan · Search · Chat — Zero Internet Required at Runtime*

**Developed by Mohammad Fayas Khan**

---

</div>

## 🧭 Overview

**MedScan+** is a fully offline, AI-powered medicine information system built with Python and Streamlit. It allows users to identify medicines by taking a picture, uploading a photo, or typing a name. It then displays detailed clinical information from a massive localized database of **11,825 medicines** and allows follow-up questions through a locally running NLP chatbot.

**Zero Internet Required.** Architecture guarantees maximum privacy. There are no external API calls, tracking pixels, or background data exfiltration.

> ✅ **Key Highlight:** Designed with a 99.1% Naive Bayes intent classification accuracy and intelligent streaming responses, featuring an ultra-fast TF-IDF indexing vectorizer.

---

## ✨ Features

| Feature | Technology Used | Description|
|---------|----------------|------------|
| 📸 **Native Camera & OCR** | OpenCV + Tesseract LSTM | Extract medicine names entirely on-device from physical pill boxes. |
| 🔍 **3-Strategy Search** | TF-IDF / Exact / Levenshtein | Intelligently falls back on fuzzy matching to compensate for spelling variations. |
| 🤖 **Offline Chatbot** | Naive Bayes + TF-IDF | A completely offline context-aware assistant with a ChatGPT-style conversational UI. |
| 📊 **Immersive UI/UX** | Custom Minimalist CSS | Premium dark mode experience designed specifically for high-density medical data. |
| 🗄️ **Massive Local DB** | Pandas + PyArrow Engine | Seamless integration with an 11.8k+ row dataset encompassing reviews, side effects, and more.|

---

## 🤖 Architecture & Under The Hood

### The 3-Strategy Search Engine

Robust searching even if the query comes from a slightly garbled OCR scan:

1. **Exact Match (O(1)):** Instant resolution if spelling is identical.
2. **TF-IDF Search:** Converts query to vector and computes cosine similarity against the corpus. Highly effective for missing keywords.
3. **Fuzzy Search:** Employs Levenshtein distance on all entries to elegantly handle minor typos.

### Intent-Driven Natural Language Chatbot

MedBot utilizes a machine learning NLP pipeline over standard logic:
- `Preprocessing:` Tokenization, lowercasing, and NLTK stopword removal.
- `Vectorization:` Cosine Similarity evaluation over robust TF-IDF vectors from a rich JSON intent file.
- `Classification:` Uses a carefully trained `sklearn` Naive Bayes classifier providing deterministic predictions.
- `Generative UI:` Responses are processed as dynamic generator streams directly into the UI components mimicking LLM token streaming but functioning offline!

---

## ⚡ Quick Start

### Requirements
- Python 3.9+
- Tesseract OCR (only required for image scanning capability)

### 1. Zero-Config Local Setup

```bash
# Clone the repository
git clone https://github.com/MohammadFayasKhan/medScan.git
cd medScan

# Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate.bat    # Windows

# Install Python packages
pip install -r requirements.txt
```

### 2. Install Tesseract (For OCR Capability)
```bash
# macOS
brew install tesseract
# Ubuntu/Debian
sudo apt install tesseract-ocr
# Windows 
# Download Installer: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Initialize Vectors (First time only)
```bash
python setup.py
```
*This handles initial dataset preprocessing, TF-IDF vector generation, and intent modeling.*

### 4. Launch Application
```bash
streamlit run app.py
```
Access MedScan+ at **`http://localhost:8501`**

---

## 🐳 Docker Deployment

The application is fully containerized. A perfect fit for isolated on-premise servers.

```bash
# Clone & Build
git clone https://github.com/MohammadFayasKhan/medScan.git
cd medScan

# Build and start via Docker Compose
docker-compose up --build
```
*Port 8501 is exposed automatically.*

---

## 🧪 Testing Coverage

Extensively tested verifying search boundaries and intent classification:
```text
======================== 66 passed in 22.38s ========================
Module                  Tests   Result   Notes
──────────────────────  ─────   ──────   ──────────────────────────
test_algorithms.py       11     ✅ Pass  NB accuracy: 99.1%
test_chatbot.py          15     ✅ Pass  All 14 intents matched
test_db.py               16     ✅ Pass  Load, validate, query
test_ocr.py              13     ✅ Pass  Text clean / extraction
test_search.py           11     ✅ Pass  Exact, TF-IDF, fuzzy
```

---

## ⚠️ Medical Disclaimer

> This software is a technological demonstration created for **educational and portfolio purposes only**. The information provided within is **not a substitute for professional medical advice**, diagnosis, or treatment.

---

## 👨‍💻 Developer

**Mohammad Fayas Khan**  
- GitHub: [@MohammadFayasKhan](https://github.com/MohammadFayasKhan)
- Built with Python · Runs 100% Offline · Docker Ready

