import torch
import unicodedata
import re

def normalize_urdu(text):
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'ے', 'ی', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()
