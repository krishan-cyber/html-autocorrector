import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    def tokenize_html(html):
        soup = BeautifulSoup(html, "html.parser")
        return " ".join([str(tag) for tag in soup.find_all()])
    
    data["incorrect_html_tokens"] = data["incorrect_html_code"].apply(tokenize_html)
    return data

def extract_features(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data["incorrect_html_tokens"])
    y = data["corrected_code"]
    return X, y, vectorizer
