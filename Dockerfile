# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — MedScan AI
# ─────────────────────────────────────────────────────────────────────────────
# This Dockerfile builds a fully self-contained image for MedScan AI.
# Once the image is built, it runs 100% offline — no internet needed at runtime.
#
# Build:  docker build -t medscan-ai .
# Run:    docker run -p 8501:8501 medscan-ai
# Or use: docker-compose up
#
# The image includes:
#   - Python 3.9 runtime
#   - All pip dependencies (streamlit, sklearn, nltk, opencv, etc.)
#   - Tesseract OCR engine (for image scanning)
#   - NLTK data downloaded at build time (punkt, stopwords, wordnet)
#   - Pre-trained ML models (Naive Bayes + TF-IDF search index)
#
# Author: Mohammad Fayas Khan
# Course: INT428 — AI Systems Design
# ─────────────────────────────────────────────────────────────────────────────

# Use official Python 3.9 slim image as base
# Slim variant is ~150MB smaller than full python:3.9
FROM python:3.9-slim

# ── Build-time labels ──────────────────────────────────────────────────────
LABEL maintainer="Mohammad Fayas Khan"
LABEL description="MedScan AI — Offline Medicine Intelligence System"
LABEL version="1.0.0"
LABEL course="INT428 — AI Systems Design"

# ── Set environment variables ──────────────────────────────────────────────
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
# Tell NLTK where to store its downloaded data inside the container
ENV NLTK_DATA=/app/nltk_data
# Streamlit config: disable usage stats and set headless mode
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ── Install system dependencies ────────────────────────────────────────────
# tesseract-ocr      : OCR engine for reading medicine labels from images
# tesseract-ocr-eng  : English language data for Tesseract
# libgl1-mesa-glx    : OpenGL library required by OpenCV
# libglib2.0-0       : GLib library required by OpenCV
# libsm6, libxext6   : Additional OpenCV runtime dependencies
# curl               : Used by Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Copy and install Python dependencies ──────────────────────────────────
# Copy requirements.txt first so Docker can cache this layer.
# If requirements.txt hasn't changed, Docker skips pip install on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy the full project into the container ───────────────────────────────
# .dockerignore prevents venv/, __pycache__/, .git/ etc. from being copied
COPY . .

# ── Run one-time setup inside the container at BUILD time ──────────────────
# This downloads NLTK data, trains the Naive Bayes classifier,
# and builds the TF-IDF search index.
# Doing this at build time means:
#   - First container startup is instant (no 5-second training wait)
#   - The container runs completely offline at runtime
RUN python setup.py

# ── Expose Streamlit port ──────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ────────────────────────────────────────────────────────────
# Docker will check this endpoint every 30 seconds.
# If it fails 3 times, the container is marked as unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Launch command ─────────────────────────────────────────────────────────
# Run Streamlit on all network interfaces (0.0.0.0) so it's accessible
# from the host machine via http://localhost:8501
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
