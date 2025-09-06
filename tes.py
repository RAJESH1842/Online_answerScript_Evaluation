import pytesseract
from PIL import Image
import difflib

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text

# Function to check for plagiarism
def check_plagiarism(text, database_texts, threshold=0.7):
    matches = []
    for db_text in database_texts:
        similarity = difflib.SequenceMatcher(None, text, db_text).ratio()
        if similarity > threshold:  # Assuming a similarity threshold of 70%
            matches.append((db_text, similarity))
    return matches

# Function to highlight keywords in the text
def highlight_keywords(text, keywords):
    highlighted_text = text
    for keyword in keywords:
        highlighted_text = highlighted_text.replace(keyword, f"{keyword}")
    return highlighted_text

# Function to grade the script based on the rubric
def grade_script(text, rubric):
    score = 0
    for criterion, points in rubric.items():
        if criterion in text:
            score += points
    return score

# End-to-End Evaluation Process
def evaluate_script(image_path, database_texts, keywords, rubric):
    # Step 1: Extract text from the scanned image
    text = extract_text_from_image(image_path)
    
    # Step 2: Check for plagiarism
    plagiarism_results = check_plagiarism(text, database_texts)
    
    # Step 3: Highlight keywords in the text
    highlighted_text = highlight_keywords(text, keywords)
    
    # Step 4: Grade the script based on the rubric
    grade = grade_script(text, rubric)
    
    return {
        "original_text": text,
        "highlighted_text": highlighted_text,
        "plagiarism_results": plagiarism_results,
        "grade": grade
    }

# Example Usage
if __name__ == "_main_":
    image_path = r"C:\testtes\img1.png"  # Path to the scanned script image
    database_texts = [
        "previously_submitted_text_1",
        "previously_submitted_text_2",
        # Add more texts as needed
    ]
    keywords = ["important_keyword_1", "important_keyword_2"]
    rubric = {
        "criterion_1": 10,
        "criterion_2": 15,
        # Add more criteria and points as needed
    }
    
    results = evaluate_script(image_path, database_texts, keywords, rubric)
    
    print("Original Text:\n", results["original_text"])
    print("Highlighted Text:\n", results["highlighted_text"])
    print("Plagiarism Results:\n", results["plagiarism_results"])
    print("Grade:\n", results["grade"])