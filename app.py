import streamlit as st
import pandas as pd
import joblib

# Page -------------------------------------------------------------------------------------------
st.set_page_config(page_title="App Review Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("📱 App Review Sentiment Analyzer")
st.markdown("Type a review and this app will predict whether it's **positive** or **negative**.")

# Model ------------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/logreg_sentiment.pkl")
    
model = load_model()

# Chat Box -----------------------------------------------------------------------------------------
with st.form("chat_form"):
    review = st.text_area("📨 Your app review:", height=150, placeholder="Write your review here...")
    submitted = st.form_submit_button("Analyze Sentiment")

# Prediction --------------------------------------------------------------------------------------
if submitted:
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
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
        pred = model.predict(input_df)[0]
        label = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(f"**Sentiment:** {label}")
        
# Dataset ----------------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_reviews.csv")

df = load_data()
        
# Most Common Words --------------------------------------------------------------------------------------------
def get_top_words(series, n=10):
    words = " ".join(series).split()
    counter = Counter(words)
    counter.pop("app", None)
    counter.pop("App", None)
    
    top = counter.most_common(n)
    return pd.DataFrame(top, columns=["Word", "Count"])

with st.expander("📄 Most Common Words by Sentiment"):
    st.markdown("**Top words in Positive Reviews**")
    st.dataframe(get_top_words(df[df.sentiment_binary == 1]["cleaned_text"]))
    st.markdown("**Top words in Negative Reviews**")
    st.dataframe(get_top_words(df[df.sentiment_binary == 0]["cleaned_text"]))


# Review Data --------------------------------------------------------------------------------------
with st.expander("🔍 Browse Reviews by Sentiment"):
    choice = st.radio("Show reviews that are...", ["All", "Positive", "Negative"])
    if choice == "All":
        st.dataframe(df[df['score'].isin([1, 2, 4, 5])][["content", "score"]].head(700))
    elif choice == "Negative":
        score_1 = df[df['score'] == 1][["content", "score"]].head(150)
        score_2 = df[df['score'] == 2][["content", "score"]].head(150)
        negative_reviews = pd.concat([score_1, score_2], ignore_index=True)
        st.dataframe(negative_reviews)
        #st.dataframe(df[df['score'].isin([1, 2])][["content", "score"]].head(250))
    elif choice == "Positive":
        score_4 = df[df['score'] == 4][["content", "score"]].head(150)
        score_5 = df[df['score'] == 5][["content", "score"]].head(150)
        positive_reviews = pd.concat([score_4, score_5], ignore_index=True)
        st.dataframe(positive_reviews)
        #st.dataframe(df[df['score'].isin([4, 5])][["content", "score"]].head(250))
