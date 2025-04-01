import easyocr
from RAG import *

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])  # Load English language model
    text = reader.readtext("resume.jpg", detail=0)  # detail=0 gives only text
    return "\n".join(text)
"""
image_text = extract_text_from_image("resume.jpg")


chat(image_text)
"""
