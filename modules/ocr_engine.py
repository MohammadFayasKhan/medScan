"""
ocr_engine.py
=============
This module handles the image → medicine name extraction pipeline.
When a user uploads a photo of a medicine package, this module processes
the image and extracts the medicine name from it.

OCR (Optical Character Recognition) pipeline steps:
  1. Convert to grayscale (colour information is not needed for text)
  2. Apply CLAHE contrast enhancement (improves readability in poor lighting)
  3. Apply adaptive thresholding (converts to binary black/white)
  4. Detect and correct image skew using Hough line transform
  5. Apply morphological operations to reduce noise
  6. Run Tesseract LSTM engine on the cleaned image
  7. Clean the raw OCR output (fix digit/letter swaps like 0→O)
  8. Extract the most likely medicine name from the recognised text

Common OCR challenges this pipeline addresses:
  - Poor lighting in photos taken with phones
  - Skewed or tilted images
  - Small text on medicine strips
  - Digit–letter confusion (e.g. "PAR4CETAM0L" vs "PARACETAMOL")

Note: Tesseract must be installed separately on the OS.
If it's not found, the module returns a helpful installation guide.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import re
import io
import logging
import numpy as np

# OpenCV — installed as opencv-python-headless (no GUI dependency)
import cv2

# Pillow for image format conversion
from PIL import Image

# pytesseract — Python wrapper around Tesseract OCR binary
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Tesseract engine/page segmentation mode:
# --oem 3  = default (LSTM neural network engine)
# --psm 6  = assume a single uniform block of text (good for labels)
OCR_CONFIG = "--oem 3 --psm 6"

# Minimum token length to be considered a candidate name
MIN_TEXT_LENGTH = 3

# Words commonly found on medicine packages that are NOT medicine names
# These are filtered out from candidate extraction
NON_MEDICINE_WORDS = {
    "sterile", "solution", "usp", "tablets", "tablet", "capsule", "capsules",
    "syrup", "injection", "ointment", "cream", "gel", "spray", "drops",
    "for", "use", "by", "date", "batch", "mfg", "exp", "lot", "no",
    "store", "below", "above", "keep", "reach", "children", "only",
    "external", "internal", "shake", "well", "before", "each", "with",
    "without", "after", "food", "water", "dose", "dosage", "maximum",
    "minimum", "adults", "adult", "children", "child", "once", "twice",
    "daily", "hours", "days", "weeks", "month", "year", "years",
    "india", "manufactured", "distributed", "marketed", "pharma",
    "pharmaceuticals", "laboratories", "labs", "pvt", "ltd", "inc",
    "not", "from", "away", "light", "heat", "cool", "dry", "place",
    "temperature", "room", "direct", "sunlight", "moisture", "air",
    "tight", "container", "pack", "package", "net", "contents", "per",
    "each", "strip", "box", "bottle", "vial", "ampoule", "sachet"
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────

def load_image_from_upload(file_obj) -> np.ndarray:
    """
    Load a PIL Image from a Streamlit file_uploader object and convert to
    an OpenCV-compatible BGR numpy array.

    Handles: jpg, jpeg, png, webp image formats.
    Converts RGBA to RGB first (OpenCV does not handle 4-channel RGBA).

    Args:
        file_obj: Streamlit UploadedFile object (BytesIO-compatible).

    Returns:
        np.ndarray: OpenCV BGR image array with shape (H, W, 3).

    Raises:
        ValueError: If the file cannot be opened as a valid image.

    Example:
        >>> # img = load_image_from_upload(uploaded_file)
        >>> # img.shape  → (480, 640, 3)
    """
    try:
        # Read file bytes — supports any format Pillow can open
        image_bytes = file_obj.read()

        # Open as PIL Image from the byte buffer
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert RGBA images to RGB (drop alpha channel)
        # OpenCV operates in BGR 3-channel; alpha must be removed first
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        # Convert PIL RGB array → numpy array → OpenCV BGR
        # PIL stores RGB; OpenCV expects BGR — flip channel order
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    except Exception as e:
        raise ValueError(f"Could not load image: {e}. Ensure the file is a valid image.")


# ─────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING STEPS
# ─────────────────────────────────────────────────────────────────────

def resize_if_needed(image: np.ndarray, min_width: int = 800) -> np.ndarray:
    """
    Scale up small images to improve OCR accuracy.

    Low-resolution images produce blurry text at pixel level which
    degrades Tesseract accuracy. Upscaling to at least 800px wide
    gives the OCR engine sufficient detail to recognise characters.

    Args:
        image (np.ndarray): Input BGR image array.
        min_width (int): Minimum width in pixels. Default 800.

    Returns:
        np.ndarray: Resized image (or original if already large enough).

    Example:
        >>> resized = resize_if_needed(small_img, min_width=800)
    """
    h, w = image.shape[:2]

    if w < min_width:
        # Compute scale factor to reach minimum width
        scale = min_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        # INTER_CUBIC provides better quality than INTER_LINEAR for upscaling
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Resized image from {w}×{h} to {new_w}×{new_h}")
        return resized

    # Image is already wide enough — return unchanged
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR colour image to single-channel grayscale.

    Grayscale reduces the image from 3 colour channels to 1 intensity channel.
    This simplifies subsequent thresholding and reduces noise sensitivity
    since colour variation between channels is not relevant for text OCR.

    Args:
        image (np.ndarray): Input BGR image array.

    Returns:
        np.ndarray: Grayscale image array with shape (H, W).

    Example:
        >>> gray = convert_to_grayscale(bgr_img)
        >>> gray.ndim == 2
        True
    """
    # Convert 3-channel BGR to single-channel grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(gray: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to reduce high-frequency noise from camera images.

    Camera sensor noise manifests as random pixel intensity variations
    that confuse edge detection and thresholding. A small Gaussian kernel
    smooths these variations while preserving larger text stroke edges.

    Args:
        gray (np.ndarray): Grayscale image array.

    Returns:
        np.ndarray: Blurred grayscale image array.

    Example:
        >>> denoised = denoise_image(gray_img)
    """
    # Gaussian kernel (3,3): small enough to preserve text edges
    # sigmaX=0 → OpenCV computes sigma automatically from kernel size
    return cv2.GaussianBlur(gray, (3, 3), 0)


def apply_threshold(denoised: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding to create a binary (black/white) image.

    Global thresholding fails on images with uneven lighting (common in
    medicine package photos). Adaptive thresholding calculates a local
    threshold for each pixel neighbourhood, handling shadows and lighting.

    Args:
        denoised (np.ndarray): Denoised grayscale image.

    Returns:
        np.ndarray: Binary image (0 or 255 per pixel).

    Example:
        >>> binary = apply_threshold(denoised_img)
    """
    return cv2.adaptiveThreshold(
        denoised,
        maxValue=255,                            # White pixel value for foreground
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Weighted Gaussian neighbourhood
        thresholdType=cv2.THRESH_BINARY,         # Standard binary output
        blockSize=11,                            # Neighbourhood size (must be odd)
        C=2                                      # Constant subtracted from mean
    )


def apply_morphology(binary: np.ndarray) -> np.ndarray:
    """
    Apply morphological closing to fill small gaps in letter strokes.

    Camera images often produce broken letter strokes in thresholded output.
    Morphological CLOSE (dilation followed by erosion) fills small holes
    and connects near-touching components, improving character connectivity.

    Args:
        binary (np.ndarray): Binary threshold image.

    Returns:
        np.ndarray: Morphologically processed binary image.

    Example:
        >>> processed = apply_morphology(binary_img)
    """
    # (2,2) kernel: small enough to avoid merging distinct letters
    kernel = np.ones((2, 2), np.uint8)

    # MORPH_CLOSE = dilate then erode: fills holes, connects nearby contours
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct image tilt (skew) using the Hough line transform.

    Tilted text confuses Tesseract's line segmentation. This function
    detects the dominant line angle in the image and counterrotates.
    Only corrects small tilts (< 15°) to avoid overcorrecting.

    Args:
        image (np.ndarray): Binary or grayscale image to deskew.

    Returns:
        np.ndarray: Deskewed image (or original if no dominant angle found).

    Example:
        >>> straight = deskew_image(tilted_img)
    """
    try:
        # Detect edges before Hough line detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Probabilistic Hough transform to detect line segments
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            # No clear lines detected — return original unchanged
            return image

        # Compute angle of each detected line segment
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # avoid division by zero for vertical lines
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)

        if not angles:
            return image

        # Use median angle to find dominant (most common) tilt direction
        median_angle = np.median(angles)

        # Only correct if tilt is small (< 15°) to avoid gross overcorrection
        if abs(median_angle) > 15:
            return image

        # Compute rotation matrix to counteract the tilt
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, scale=1.0)

        # Apply affine rotation with white border fill (255 = white background)
        deskewed = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )
        return deskewed

    except Exception as e:
        # If deskewing fails for any reason, return original unchanged
        logger.warning(f"Deskew failed: {e} — returning original")
        return image


# ─────────────────────────────────────────────────────────────────────
# OCR EXECUTION
# ─────────────────────────────────────────────────────────────────────

def run_tesseract(preprocessed: np.ndarray) -> str:
    """
    Run Tesseract OCR on a preprocessed grayscale image.

    Uses the configured OCR_CONFIG (OEM 3, PSM 6) for maximum accuracy
    on uniform text blocks as found on medicine labels.

    Args:
        preprocessed (np.ndarray): Preprocessed (binary) image array.

    Returns:
        str: Raw OCR text output from Tesseract.

    Raises:
        RuntimeError: If Tesseract is not installed (with install guide).

    Example:
        >>> text = run_tesseract(preprocessed_img)
        >>> len(text) > 0
        True
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract not installed. Run: pip install pytesseract"
        )

    try:
        # Pass processed image to Tesseract with configured options
        raw_text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG)
        return raw_text
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR binary not found on your system.\n"
            "Install Tesseract:\n"
            "  Mac:     brew install tesseract\n"
            "  Linux:   sudo apt install tesseract-ocr\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )


# ─────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────

def clean_ocr_output(raw_text: str) -> str:
    """
    Normalise and clean raw OCR text output for medicine name extraction.

    Common OCR errors on medicine labels:
    - Extra whitespace between characters
    - Non-printable control characters
    - Digit/letter confusions (0→O, 1→I, 5→S in all-caps tokens)

    Args:
        raw_text (str): Raw string returned by Tesseract.

    Returns:
        str: Cleaned, normalised OCR text string.

    Example:
        >>> clean_ocr_output("  PARACETAM0L\n\n5OOmg  ")
        'PARACETAMOL 500mg'
    """
    if not raw_text:
        return ""

    # Step 1: Strip leading/trailing whitespace
    text = raw_text.strip()

    # Step 2: Remove non-printable characters (control chars, null bytes)
    # Keep: printable ASCII only (codes 32–126)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Step 3: Collapse multiple whitespace into single space per line
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]

    # Step 4: Fix common OCR digit→letter confusions in all-caps tokens
    corrected_lines = []
    for line in lines:
        if not line:
            continue
        tokens = line.split()
        corrected_tokens = []
        for token in tokens:
            # Only apply digit corrections to tokens that look like drug names
            # (predominantly uppercase letters with occasional digits)
            if len(token) > 3 and any(c.isalpha() for c in token):
                upper_ratio = sum(1 for c in token if c.isupper()) / len(token)
                if upper_ratio > 0.6:
                    # Likely an all-caps drug name token — apply corrections
                    token = token.replace("0", "O").replace("1", "I").replace("5", "S")
            corrected_tokens.append(token)
        corrected_lines.append(" ".join(corrected_tokens))

    # Step 5: Rejoin lines and normalise final whitespace
    cleaned = " ".join(corrected_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_candidates(cleaned_text: str) -> list:
    """
    Extract likely medicine name candidates from cleaned OCR text.

    Filters tokens using multiple heuristics:
    - Length > MIN_TEXT_LENGTH characters
    - Not in NON_MEDICINE_WORDS blocklist
    - Not purely numeric (e.g., "500")
    - Not all special characters

    Sorts by token length descending (longer tokens more likely to be brand names).

    Args:
        cleaned_text (str): Cleaned OCR output string.

    Returns:
        list[str]: Up to 5 most likely medicine name candidate strings.

    Example:
        >>> extract_candidates("PARACETAMOL 500mg Store below 25C")
        ['PARACETAMOL']
    """
    if not cleaned_text:
        return []

    # Step 1: Split text on whitespace and newlines
    tokens = re.split(r"[\s\n/,;:()]+", cleaned_text)

    candidates = []
    for token in tokens:
        # Step 2: Filter by minimum length
        if len(token) <= MIN_TEXT_LENGTH:
            continue

        # Step 3: Filter against non-medicine word blocklist (case-insensitive)
        if token.lower() in NON_MEDICINE_WORDS:
            continue

        # Step 4: Filter out purely numeric tokens (e.g., "500", "100")
        if token.isdigit():
            continue

        # Step 5: Filter out tokens consisting only of special characters
        if not any(c.isalpha() for c in token):
            continue

        # Step 6: Filter out tokens containing only punctuation or numbers
        alpha_ratio = sum(1 for c in token if c.isalpha()) / len(token)
        if alpha_ratio < 0.5:
            # More than half the characters are non-alphabetic → likely not a name
            continue

        candidates.append(token)

    # Step 7: Sort by length descending (longer = more likely to be a full drug name)
    candidates.sort(key=len, reverse=True)

    # Remove duplicates while maintaining order (case-insensitive dedup)
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique_candidates.append(c)

    # Return top 5 candidates maximum
    return unique_candidates[:5]


# ─────────────────────────────────────────────────────────────────────
# MAIN OCR PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_ocr_pipeline(file_obj) -> dict:
    """
    Run the complete OCR pipeline end-to-end on an uploaded image file.

    Pipeline stages:
      1. Load image from upload bytes
      2. Resize if too small
      3. Convert to grayscale
      4. Denoise with Gaussian blur
      5. Adaptive threshold to binary
      6. Morphological close to fix broken strokes
      7. Deskew to correct tilt
      8. Run Tesseract OCR
      9. Clean OCR output
      10. Extract name candidates

    Args:
        file_obj: Streamlit UploadedFile or BytesIO-compatible object.

    Returns:
        dict: Comprehensive pipeline result containing:
            {
              "success": bool,
              "raw_text": str,
              "cleaned_text": str,
              "candidates": list[str],
              "best_candidate": str,
              "preprocessed_image": np.ndarray,
              "original_image": np.ndarray,
              "error": str | None
            }

    Example:
        >>> result = run_ocr_pipeline(uploaded_file)
        >>> result["success"]
        True
        >>> result["best_candidate"]
        'PARACETAMOL'
    """
    # Initialise result dict with safe defaults
    result = {
        "success": False,
        "raw_text": "",
        "cleaned_text": "",
        "candidates": [],
        "best_candidate": "",
        "preprocessed_image": None,
        "original_image": None,
        "error": None
    }

    try:
        # ── Stage 1: Load image from uploaded file bytes ──────────────
        image = load_image_from_upload(file_obj)
        result["original_image"] = image.copy()  # preserve original for display

        # ── Stage 2: Resize small images for better OCR accuracy ──────
        image = resize_if_needed(image, min_width=800)

        # ── Stage 3: Convert colour image to grayscale ────────────────
        gray = convert_to_grayscale(image)

        # ── Stage 4: Reduce high-frequency noise with Gaussian blur ───
        denoised = denoise_image(gray)

        # ── Stage 5: Create binary image via adaptive threshold ────────
        binary = apply_threshold(denoised)

        # ── Stage 6: Close small gaps in letter strokes ───────────────
        processed = apply_morphology(binary)

        # ── Stage 7: Correct image tilt via Hough line analysis ───────
        processed = deskew_image(processed)

        # Store final preprocessed image for side-by-side UI display
        result["preprocessed_image"] = processed

        # ── Stage 8: Run Tesseract OCR on preprocessed image ──────────
        raw_text = run_tesseract(processed)
        result["raw_text"] = raw_text

        # ── Stage 9: Normalise and fix common OCR text errors ─────────
        cleaned = clean_ocr_output(raw_text)
        result["cleaned_text"] = cleaned

        # ── Stage 10: Extract top medicine name candidates ─────────────
        candidates = extract_candidates(cleaned)
        result["candidates"] = candidates

        # Best candidate is the first (longest alphabetic) token
        result["best_candidate"] = candidates[0] if candidates else ""

        # Mark pipeline as successful
        result["success"] = True

        logger.info(f"OCR pipeline complete. Best candidate: '{result['best_candidate']}'")

    except RuntimeError as e:
        # Tesseract not installed — specific actionable error
        result["error"] = str(e)
        logger.error(f"OCR Runtime error: {e}")

    except ValueError as e:
        # Invalid image file format
        result["error"] = str(e)
        logger.error(f"OCR image load error: {e}")

    except Exception as e:
        # Catch-all for unexpected errors
        result["error"] = f"Unexpected OCR error: {e}"
        logger.error(f"Unexpected OCR error: {e}", exc_info=True)

    return result
