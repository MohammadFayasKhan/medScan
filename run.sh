#!/bin/bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MEDSCAN AI — ANTIGRAVITY BUILD"
echo "  Offline Medicine Intelligence System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "[2/4] Running setup (NLTK + model training)..."
python setup.py
echo ""
echo "[3/4] Launching MEDSCAN AI..."
echo "  Open your browser at: http://localhost:8501"
echo ""
streamlit run app.py
