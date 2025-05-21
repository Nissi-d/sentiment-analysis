import json
import os
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load data
df = pd.read_csv("data/processed/preprocessed_reviews.csv")
X = df[["cleaned_text", "exclamation_count", "question_count", "has_upvotes"]]
y = df["sentiment_binary"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. ColumnTransformer that keeps everything ≥ 0
preprocessor = ColumnTransformer([
    # TF-IDF on text → always ≥0
    ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"), "cleaned_text"),
    # Rescale numeric counts to [0,1]
    ("num", MinMaxScaler(), ["exclamation_count", "question_count", "has_upvotes"])
])

# 4. Pipeline with MultinomialNB
nb_pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", MultinomialNB(alpha=1.0))  # alpha is your Laplace smoothing
])

# 5. Train
nb_pipeline.fit(X_train, y_train)



# save model
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "nb_sentiment.pkl")
joblib.dump(nb_pipeline, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")


# test
y_pred = nb_pipeline.predict(X_test)

# evaluation metrics
acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=["Neg","Pos"], digits=4))

# create evaluation directory
os.makedirs("evaluation", exist_ok=True)

# save numeric metrics to JSON
metrics = {
    "model": "naive bayes",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
}
with open("evaluation/naivebayes_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)