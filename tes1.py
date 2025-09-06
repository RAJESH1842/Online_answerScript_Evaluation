import pytesseract
from PIL import Image
import difflib

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

def extract_text_from_image(image_path):

    image = Image.open(image_path)
    
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


def highlight_keywords(text, keywords):
    highlighted_text = text
    for keyword in keywords:
        highlighted_text = highlighted_text.replace(keyword, f"**{keyword}**")
    return highlighted_text


def grade_script(text, rubric):
    score = 0
    for criterion, points in rubric.items():
        if criterion in text:
            score += points
    return score


def evaluate_script(image_path, database_texts, keywords, rubric):
    
    text = extract_text_from_image(image_path)
    
    plagiarism_results = check_plagiarism(text, database_texts)
     
    highlighted_text = highlight_keywords(text, keywords)
    
    
    grade = grade_script(text, rubric)
    
    return {
        "original_text": text,
        "highlighted_text": highlighted_text,
        "plagiarism_results": plagiarism_results,
        "grade": grade
    }

# Example Usage

if __name__ == "__main__":
    image_path = r"C:\testtes\img1.png" # Path to the scanned script image
    database_texts = [
        "previously_submitted_text_1",
        "previously_submitted_text_2",
        # Add more texts as needed
    ]
    keywords = ["important_keyword_1", "important_keyword_2"]
    rubric = {
        "criterion_1": 10,
        "criterion_2": 15,
        
    }
    
    results = evaluate_script(image_path, database_texts, keywords, rubric)
    
    print("Original Text:\n", results["original_text"])
    print("Highlighted Text:\n", results["highlighted_text"])
    print("Plagiarism Results:\n", results["plagiarism_results"])
    print("Grade:\n", results["grade"])
