import sys
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Update with your tesseract.exe path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def run_ocr_on_image(image_path):
    print(f"\nRunning OCR on: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Image not loaded.")
        return

    # Preprocessing for better accuracy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)

    # OCR with Tesseract
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=config)

    print("\n✅ OCR Result:\n")
    print(text)

def run_ocr_on_pdf(pdf_path):
    print(f"\nConverting PDF to images: {pdf_path}")
    pages = convert_from_path(pdf_path)

    for i, page in enumerate(pages):
        temp_img = f"page_{i+1}.png"
        page.save(temp_img, "PNG")
        run_ocr_on_image(temp_img)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_final.py <image_or_pdf_path>")
    else:
        file_path = sys.argv[1]
        if file_path.lower().endswith(".pdf"):
            run_ocr_on_pdf(file_path)
        else:
            run_ocr_on_image(file_path)
