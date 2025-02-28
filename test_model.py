import pickle
from preprocess import preprocess_data, extract_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from bs4 import BeautifulSoup
import pandas as pd

def load_model(model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def evaluate_model(model, vectorizer, test_data_path):
   
    
    test_data = pd.read_csv(test_data_path)
    test_data = preprocess_data(test_data)
    X_test, y_test, _ = extract_features(test_data)
    
    
    y_pred = model.predict(X_test)
    
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    


test_data_path = "data/labeled.csv"  
    

model, vectorizer = load_model()
    

evaluate_model(model, vectorizer, test_data_path)
    