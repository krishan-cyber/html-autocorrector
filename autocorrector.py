import pickle
from preprocess import preprocess_data

def load_model_and_vectorizer(model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def autocorrect_html(model, vectorizer, incorrect_html):
    from bs4 import BeautifulSoup

    def tokenize_html(html):
        soup = BeautifulSoup(html, "html.parser")
        return " ".join([str(tag) for tag in soup.find_all()])

    incorrect_html_tokens = tokenize_html(incorrect_html)
    incoorect_html__vector = vectorizer.transform([incorrect_html_tokens])
    corrected_html = model.predict(incoorect_html__vector)
    return corrected_html[0]


model, vectorizer = load_model_and_vectorizer()


incorrect_html = "<div><i>Improperly nested</b></i></div>"
corrected_html = autocorrect_html(model, vectorizer, incorrect_html)


print("Original HTML:", incorrect_html)
print("Corrected HTML:", corrected_html)
