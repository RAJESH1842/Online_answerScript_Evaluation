# ocr_improved.py
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract

# set Tesseract path (edit if different on your PC)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# try import easyocr if installed
try:
    import easyocr
    EASY = True
except Exception:
    EASY = False

def safe_avg_conf(conf_list):
    # keep only 0..100 ints
    vals = [int(c) for c in conf_list if c.isdigit() or (c.lstrip('-').isdigit() and int(c) >= 0)]
    vals = [v for v in vals if 0 <= v <= 100]
    return sum(vals)/len(vals) if vals else 0.0

def tesseract_data(pil_img, psm=7, lang='eng'):
    cfg = f'--oem 1 --psm {psm} -l {lang}'
    data = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
    words = [w for w in data['text'] if w.strip()]
    text = " ".join(words)
    confs = data.get('conf', [])
    avg_conf = safe_avg_conf([str(c) for c in confs])
    return text, avg_conf, data

def easyocr_recognize(cv_bgr):
    if not EASY:
        return "[EasyOCR not installed]", 0.0
    reader = easyocr.Reader(['en'], gpu=False)
    img_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
    res = reader.readtext(img_rgb, detail=0, paragraph=True)
    text = "\n".join(res)
    # EasyOCR does not return a simple average confidence from this call; return 0 for simplicity
    return text, 0.0

# basic preprocessing for line segmentation
def preprocess_for_lines(cv_bgr, scale=2.0):
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
    # increase contrast and threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.medianBlur(gray, 3)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,9)
    # invert for projection (text=1)
    inv = 255 - th
    return th, inv

# segment lines using horizontal projection
def segment_lines(inv_img):
    # inv_img should be binary with text=255 background=0
    # Project along rows
    hist = np.sum(inv_img, axis=1)
    # threshold to detect text rows
    h_thresh = (np.max(hist) * 0.02)
    lines = []
    in_line = False
    start = 0
    for i, v in enumerate(hist):
        if v > h_thresh and not in_line:
            in_line = True
            start = i
        elif v <= h_thresh and in_line:
            in_line = False
            end = i
            if end - start > 6:
                lines.append((start, end))
    # handle if ends in line
    if in_line:
        lines.append((start, len(hist)-1))
    return lines

def run_all(path_img):
    cv_img = cv2.imread(path_img)
    if cv_img is None:
        print("Cannot read image:", path_img); return
    print("Image loaded:", path_img)
    # 1) Full-image Tesseract (psm 6 & 11)
    pil_full = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    ttext6, conf6, _ = tesseract_data(pil_full, psm=6)
    ttext11, conf11, _ = tesseract_data(pil_full, psm=11)
    print("\nTESSERACT full-image psm=6 conf:", conf6)
    print(ttext6)
    print("\nTESSERACT full-image psm=11 conf:", conf11)
    print(ttext11)

    # 2) EasyOCR (if present)
    if EASY:
        easy_text, _ = easyocr_recognize(cv_img)
        print("\nEasyOCR full-image result:\n", easy_text)
    else:
        print("\nEasyOCR not installed - skip.")

    # 3) Line segmentation + per-line Tesseract (often better for handwriting)
    th, inv = preprocess_for_lines(cv_img, scale=2.0)
    lines = segment_lines(inv)
    print(f"\nDetected {len(lines)} lines (line segmentation). Running per-line OCR (psm=7):")
    all_lines = []
    avg_confs = []
    for idx, (s,e) in enumerate(lines):
        # crop with small padding
        pad = 6
        s2 = max(0, s-pad); e2 = min(inv.shape[0]-1, e+pad)
        crop = th[s2:e2, :]
        pil_crop = Image.fromarray(crop)
        text, conf, _ = tesseract_data(pil_crop, psm=7)
        print(f"\nLINE {idx+1} (rows {s2}:{e2}) conf={conf:.2f}\n{text}")
        all_lines.append(text)
        avg_confs.append(conf)
    if all_lines:
        print("\nMerged per-line text:\n", "\n".join(all_lines))
        print("Average line conf:", (sum(avg_confs)/len(avg_confs)) if avg_confs else 0.0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_improved.py <image_path>")
    else:
        run_all(sys.argv[1])
