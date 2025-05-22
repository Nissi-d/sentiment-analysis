from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model/logreg_sentiment.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        review = request.form.get("input")

        # Build the required input format
        input_data = {
            'cleaned_text': [review],
            'has_upvotes': [0],
            'review_length': [len(review.split())],
            'exclamation_count': [review.count('!')],
            'question_count': [review.count('?')],
            'ngram_great app': [0],
            'ngram_good app': [0],
            'ngram_easy use': [0],
            'ngram_love app': [0],
            'ngram_pro version': [0],
            'ngram_google calendar': [0],
            'ngram_free version': [0],
            'ngram_use app': [0],
            'ngram_like app': [0],
            'ngram_doesnt work': [0],
            'ngram_really like app': [0],
            'ngram_app easy use': [0],
            'ngram_buy pro version': [0],
            'ngram_using app years': [0],
            'ngram_paid pro version': [0],
            'ngram_really good app': [0],
            'ngram_simple easy use': [0],
            'ngram_used app years': [0],
            'ngram_sync google calendar': [0],
            'ngram_todo list app': [0]
        }

        input_df = pd.DataFrame(input_data)
        prediction = model.predict(input_df)[0]
        label = "Positive" if prediction == 1 else "Negative"

        return render_template("result.html", prediction=label)

    return "Something went wrong."
