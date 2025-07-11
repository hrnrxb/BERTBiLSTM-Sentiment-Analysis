
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)   
    text = re.sub(r"\s+", " ", text).strip()     
    return text
