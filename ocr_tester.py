# ocr_tester.py
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

# <-- EDIT: set your tesseract exe path if needed -->
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------
# Preprocessing functions
# ------------------------
def preprocess_basic(img_bgr, scale=2.0, clahe=True, denoise=True, adaptive_thresh=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe_obj.apply(gray)
    if adaptive_thresh:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 9)
    # small closing to fill strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    gray = deskew(gray)
    return gray

def deskew(img):
    coords = np.column_stack(np.where(img < 255))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.1:
        return img
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated

# ------------------------
# OCR helpers
# ------------------------
def tesseract_ocr_with_confidence(pil_img, psm=6, oem=1, lang='eng'):
    config = f'--oem {oem} --psm {psm} -l {lang} -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.:;!?\'"()@/\\-'
    # image_to_data returns box/word-level confidences
    data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
    text = " ".join([w for w in data['text'] if w.strip()!=''])
    # confidences are strings; -1 means no confidence
    confs = [int(c) for c in data['conf'] if c != '-1']
    avg_conf = float(sum(confs))/len(confs) if confs else 0.0
    return text, avg_conf, data

# ------------------------
# Multi-run tester
# ------------------------
def run_tests(image_path):
    print("Loading:", image_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("ERROR: could not read image.")
        return

    # try a few preprocessing combos and psm values
    combos = [
        {"scale":2.0, "clahe":True, "denoise":True, "adaptive":True},
        {"scale":3.0, "clahe":True, "denoise":False, "adaptive":True},
        {"scale":1.5, "clahe":False, "denoise":True, "adaptive":True},
        {"scale":2.0, "clahe":True, "denoise":True, "adaptive":False}
    ]
    psm_list = [6,7,11,3,4]  # try several PSMs

    results = []
    for c in combos:
        pre = preprocess_basic(img_bgr, scale=c["scale"], clahe=c["clahe"], denoise=c["denoise"], adaptive_thresh=c["adaptive"])
        pil = Image.fromarray(pre)
        for psm in psm_list:
            text, conf, _ = tesseract_ocr_with_confidence(pil, psm=psm)
            results.append({
                "combo": c,
                "psm": psm,
                "avg_conf": conf,
                "text": text
            })
            print(f"PSM={psm} scale={c['scale']} clahe={c['clahe']} denoise={c['denoise']} adaptive={c['adaptive']} => avg_conf={conf:.2f}")
            print("---- TEXT START ----")
            print(text)
            print("---- TEXT END ----\n")

    # sort by avg_conf descending and print top 3
    results_sorted = sorted(results, key=lambda r: r['avg_conf'], reverse=True)
    print("\nTop 3 results by average confidence:")
    for r in results_sorted[:3]:
        print(f"PSM={r['psm']} scale={r['combo']['scale']} avg_conf={r['avg_conf']:.2f}\nText:\n{r['text']}\n---\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_tester.py <image_path>")
    else:
        run_tests(sys.argv[1])
