import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv(os.path.join("data","processed","preprocessed_reviews.csv"))

text_col = 'cleaned_text'
numeric_features = ['exclamation_count', 'question_count', 'has_upvotes']

X = df[[text_col] + numeric_features]
y = df['sentiment_binary']

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# build pipeline
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english'), text_col),
    ("num", StandardScaler(), numeric_features)
])

pipeline = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# train
pipeline.fit(X_train, y_train)

# save model
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "logreg_sentiment.pkl")
joblib.dump(pipeline, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")

# test
y_pred = pipeline.predict(X_test)

# evaluation metrics
acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
clf_rep = classification_report(
    y_test, y_pred,
    target_names=["Negative", "Positive"],
    digits=4
)

# create evaluation directory
os.makedirs("evaluation", exist_ok=True)

# save numeric metrics to JSON
metrics = {
    "model": "logreg",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
}
with open("evaluation/logreg_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg","Pos"],
            yticklabels=["Neg","Pos"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LogReg Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation/logreg_confusion_matrix.png")
plt.close()

print("Evaluation metrics & plots saved to /evaluation/")