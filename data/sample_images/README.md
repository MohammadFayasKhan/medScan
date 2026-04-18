# Sample Images for MEDSCAN AI OCR Testing

Place medicine package images (JPG, PNG, WEBP) in this directory to test the OCR scanner.

## Recommended Test Images

- Clear photographs of medicine strip/blister packs
- Medicine box/carton front face
- Prescription label images

## Tips for Good OCR Results

1. **Lighting**: Use natural light or bright indoor lighting. Avoid harsh shadows.
2. **Focus**: Ensure the text is in sharp focus — blurry images give poor OCR results.
3. **Angle**: Keep the camera parallel to the label (avoid tilted shots).
4. **Size**: Higher resolution images (min 800px wide) give better results.
5. **Contrast**: High-contrast labels (dark text on white/light background) work best.

## How to Add Images

1. Take a clear photo of the medicine package
2. Save as: `medicine_name.jpg` (e.g., `paracetamol_strip.jpg`)
3. Place in this directory
4. Launch app → Tab 1 → Upload Image → Select your file

## Acceptable Formats

JPG, JPEG, PNG, WEBP

## Note on Tesseract

The OCR engine uses Tesseract. If Tesseract is not installed, the Upload Image mode
will show an error with installation instructions. You can still use the "Type Name"
mode to search the database without Tesseract installed.

Install Tesseract:
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
