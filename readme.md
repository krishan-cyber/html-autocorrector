HTML Autocorrector

The HTML Autocorrector is a machine learning-based tool that automatically identifies and corrects common bad coding practices in HTML code. The tool uses a Random Forest classifier trained on labeled examples of bad and corrected HTML code. It supports evaluating model performance, training with new data, and testing with specific examples.

File hierarchy

html autocorrector/
│
├── data/
│   
│   ├── labeled_data.csv          # Labeled dataset (incoorect_html_code and corrected_code)
│
├── preprocess.py                 # Script for preprocessing and feature extraction
├── train_model.py                # Script for training the Random Forest model
├── test_model.py                 # Script for model evaluation and testing
├── autocorrector.py              # Script for using the trained model to correct HTML
│
├── model.pkl                     # Pre-trained Random Forest model
├── vectorizer.pkl                # Vectorizer used for tokenizing HTML
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation


MODEL DEVLOPMENT:

Data Preparation:
Dataset choosen here is anotated and labeled by human expert.However in future this can be and going to be done using machine learning
it contains incorrect html code and related correct html code.

Preprocessing:
The incorrect HTML was tokenized using BeautifulSoup and converted into numerical vectors with CountVectorizer. The corrected HTML was used as the target labels.

Model Prediction:
model used here for prediction is  Random Forest model, to predict the corrected HTML for the incorrect HTML snippets.
Random Forest is chosen because it provides good performance even with relatively small datasets.and is less prone to overfitting

Random Forest Hyperparameters:
n_estimators: 100
random_state: 42
criterion: "gini"
max_depth: None (default)


Evaluation:
The model's predictions were compared to the actual corrected HTML in the test set using scikit-learn's evaluation metrics:
accuracy_score for overall correctness.
precision_score, recall_score, and f1_score for deeper insights into how well the model performs.

with above dataset ,my model get good accuracy here are results:
Accuracy: 0.8000
Precision: 0.7000
Recall: 0.8000
F1-Score: 0.7333

however i believe with more data this model can perform much better and can give high accuracy.


