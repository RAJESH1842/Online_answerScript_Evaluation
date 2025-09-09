import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pdf2image import convert_from_path
from PIL import Image

# Update this path to where your Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def run_ocr(file_path):
    text = ""
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, 300)
        for i, page in enumerate(pages):
            gray = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
            text += pytesseract.image_to_string(page, config="--psm 6")
    else:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 6")
    return text

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Images and PDF", "*.png;*.jpg;*.jpeg;*.pdf")])
    if not file_path:
        return
    result = run_ocr(file_path)
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, result)

def save_text():
    text = text_box.get(1.0, tk.END).strip()
    if not text:
        messagebox.showwarning("Empty", "No text to save!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo("Saved", f"Text saved to {file_path}")

# GUI
root = tk.Tk()
root.title("OCR Extractor")
root.geometry("600x500")

btn_open = tk.Button(root, text="📂 Open Image/PDF", command=open_file)
btn_open.pack(pady=10)

text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
text_box.pack(pady=10)

btn_save = tk.Button(root, text="💾 Save as TXT", command=save_text)
btn_save.pack(pady=10)

root.mainloop()
