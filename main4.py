from tkinter import *
from tkinter import filedialog
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Global variable to store recognized text
recognized_text = None

def browseFiles():
    global recognized_text
    filetypes = (("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp *.gif"), ("PDF files", "*.pdf"), ("All files", "*.*"))
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=filetypes)
    
    if filename == "":
        return
    
    # Recognize text from image or PDF
    recognized_text = recognize_text(filename)
    
    # Display recognized text
    label_file_explorer.configure(text=recognized_text)

def recognize_text(filename):
    # Check if file is a PDF
    if filename.lower().endswith('.pdf'):
        return recognize_text_from_pdf(filename)
    else:
        return recognize_text_from_image(filename)

def recognize_text_from_image(filename):
    # Load image with OpenCV
    img = cv2.imread(filename)
    
    # Preprocess image: convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance image contrast
    img_enhanced = enhance_contrast(img_gray)
    
    # Apply Gaussian blur to smoothen the image
    img_blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    
    # Use adaptive thresholding to binarize image
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Perform noise reduction
    kernel = np.ones((1, 1), np.uint8)
    img_processed = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    
    # Recognize text using Tesseract OCR
    custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    recognized_text = pytesseract.image_to_string(img_processed, config=custom_config)
    
    return recognized_text

def enhance_contrast(image):
    # Enhance image contrast using PIL's ImageEnhance module
    image_pil = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(image_pil)
    enhanced_img = enhancer.enhance(2.0)  # Increase contrast (adjust as needed)
    # Apply sharpening filter
    enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
    return np.array(enhanced_img)

def recognize_text_from_pdf(filename):
    # Recognize text from PDF using Tesseract OCR
    recognized_text = pytesseract.image_to_string(filename, config='--oem 3 --psm 6')
    return recognized_text

def pdf():
    global recognized_text
    if recognized_text:
        pdf_filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if pdf_filename:
            with open(pdf_filename, 'w', encoding="utf-8") as f:
                f.write(recognized_text)

# Create the main window
window = Tk()
window.title('Handwritten Text Recognition')
window.geometry("700x350")
window.config(background="white")

# Add labels and buttons to the window
reg_info = Label(window, text="Handwritten Text Recognition Using Pytesseract", width=80, height=2, font=("Arial", 12, "bold"), fg="black", bg="lightgrey")
reg_info.place(x=370, y=18, anchor='center')

label_file_explorer = Label(window, text="See the Output Here", font=("Arial", 10, "bold"), width=90, height=12, fg="blue")
label_file_explorer.place(x=0, y=35)

button_explore = Button(window, text="Browse Files", fg="white", bg="black", font=("Arial", 10, "bold"), width=10, command=browseFiles)
button_explore.place(x=250, y=270)

text_label = Label(window, text="(Select an image or PDF)", bg="white", fg="black", font=("Arial", 8, "bold"))
text_label.place(x=242, y=300)

button_pdf = Button(window, text="Save as PDF", fg="white", bg="black", font=("Arial", 10, "bold"), width=15, command=pdf)
button_pdf.place(x=370, y=270)

window.mainloop()
