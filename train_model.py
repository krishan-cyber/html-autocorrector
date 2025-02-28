import pickle
import pandas as pd
from preprocess import preprocess_data, extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model(data_path, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    
    data = pd.read_csv(data_path)
    data = preprocess_data(data)
    X, y, vectorizer = extract_features(data)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model and vectorizer saved to {model_path} and {vectorizer_path}.")


train_and_save_model("data/labeled.csv")
