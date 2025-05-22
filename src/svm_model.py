import json
import os
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import LinearSVC
import seaborn as sns

# load 
df = pd.read_csv("data/processed/preprocessed_reviews.csv")
X = df[["cleaned_text", "exclamation_count", "question_count", "has_upvotes"]]
y = df["sentiment_binary"]

# training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# naivebayes requires that text keeps everything ≥ 0
preprocessor = ColumnTransformer([
    # TF-IDF on text always ≥0
    ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"), "cleaned_text"),
    # rescale numeric counts to [0,1]
    ("num", MinMaxScaler(), ["exclamation_count", "question_count", "has_upvotes"])
])

#SVM model training and testing
svm_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LinearSVC(max_iter=10_000))
])

#training
svm_pipeline.fit(X_train, y_train)
#testing
y_pred_svm = svm_pipeline.predict(X_test)

os.makedirs("models", exist_ok=True)

MODEL_PATH = os.path.join("models", "svm_sentiment.pkl")
joblib.dump(svm_pipeline, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")

# evaluation metrics
acc_svm   = accuracy_score(y_test, y_pred_svm)
prec_svm  = precision_score(y_test, y_pred_svm)
rec_svm   = recall_score(y_test, y_pred_svm)
f1_svm    = f1_score(y_test, y_pred_svm)
print(classification_report(y_test, y_pred_svm, target_names=["Neg","Pos"], digits=4))

# create evaluation directory
os.makedirs("evaluation", exist_ok=True)

# save numeric metrics to JSON
metrics = {
    "model": "SVM",
    "accuracy": acc_svm,
    "precision": prec_svm,
    "recall": rec_svm,
    "f1": f1_svm
}
with open("evaluation/svm_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

# confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg","Pos"],
            yticklabels=["Neg","Pos"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation/svm_confusion_matrix.png")
plt.close()