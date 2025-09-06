import cv2
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from PIL import Image
import pytesseract
import keras_ocr

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove noise
    noise_removed_image = cv2.medianBlur(binary_image, 3)
    
    return noise_removed_image

# Function to perform OCR using Tesseract
def ocr_image(preprocessed_image):
    # Convert image to PIL format
    pil_image = Image.fromarray(preprocessed_image)
    
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(pil_image)
    
    return text

# Function to perform OCR using keras-ocr
def advanced_ocr(image_path):
    # Create a pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    
    # Read the image
    image = keras_ocr.tools.read(image_path)
    
    # Perform OCR
    prediction_groups = pipeline.recognize([image])
    
    # Extract text from the predictions
    text = ' '.join([word for box, word in prediction_groups[0]])
    
    return text

# Main function for basic OCR
def main_basic(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Perform OCR
    text = ocr_image(preprocessed_image)
    
    print("Extracted Text (Basic):")
    print(text)

# Main function for advanced OCR
def main_advanced(image_path):
    # Perform advanced OCR
    text = advanced_ocr(image_path)
    
    print("Extracted Text (Advanced):")
    print(text)

# Run the main function with the path to your image
if __name__ == "__main__":
    image_path = 'C:/testtes/img2.jpg'
    
    # Uncomment one of the following lines to use the respective OCR method
    # main_basic(image_path)  # For basic OCR using Tesseract
    main_advanced(image_path)  # For advanced OCR using keras-ocr
