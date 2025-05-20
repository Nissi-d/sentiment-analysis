import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


nltk.download("stopwords")
nltk.download("punkt")

# load dataset
df = pd.read_csv("data/raw/reviews.csv") 
print("Initial shape of data:", df.shape)

# check for missing values
print("\nMissing values by column:")
print(df.isnull().sum())

# create the binary sentiment labels from star rating 
df['sentiment_binary'] = np.nan
df.loc[df['score'].isin([1, 2]), 'sentiment_binary'] = 0    # 1–2 stars -> negative(0)
df.loc[df['score'].isin([4, 5]), 'sentiment_binary'] = 1    # 4–5 stars -> positive(1)

# drop 3-star ratings
df = df.dropna(subset=['sentiment_binary'])  
print("After label assignment:", df['sentiment_binary'].value_counts())

text_column = 'content'  

# stopwords for text preprocessing
stop_words = set(stopwords.words("english"))

# function to clean a review
def clean(text):
    text = text.lower()                         # make all text lowercase
    text = re.sub(r'[^a-z\s]', '', text)        # remove special chars, punctuation, numbers
    tokens = text.split()                       # split text by words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# clean all reviews in column
df['cleaned_text'] = df[text_column].astype(str).apply(clean)
print("\nSample cleaned text:")
print(df['cleaned_text'].head(5))

# remove empty rows after cleaning
df = df[df['cleaned_text'].str.strip() != '']
print("\nAfter removing empty cleaned texts:", df.shape)

# FEATURE ENGINEERING: 

# reviews with upvotes
df['has_upvotes'] = (df['thumbsUpCount'] > 0).astype(int)
# word count (cleaned reviews)
df['review_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
# exclamation count (from raw data)
df['exclamation_count'] = df[text_column].astype(str).apply(lambda x: x.count('!'))
# question mark count (in each review)
df['question_count'] = df[text_column].astype(str).apply(lambda x: x.count('?'))

# n-grams in review    
def add_top_ngrams(df, text_column, top_k=10, ngram_range=(2, 2)):
    # create n-gram vectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(df[text_column])

    # total frequency of each n-gram
    total_counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    ngram_freq = sorted(zip(vocab, total_counts), key=lambda x: x[1], reverse=True)

    # Keep only top N
    top_ngrams = [ngram for ngram, count in ngram_freq[:top_k]]

    print(f"Top {top_k} n-grams:", top_ngrams)

    # new vectorizer for those top n-grams
    selected_vectorizer = CountVectorizer(vocabulary=top_ngrams, binary=True)
    X_selected = selected_vectorizer.transform(df[text_column])

    # new df
    ngram_df = pd.DataFrame(X_selected.toarray(), columns=[f'ngram_{n}' for n in top_ngrams])

    # concatenate 
    df = pd.concat([df.reset_index(drop=True), ngram_df.reset_index(drop=True)], axis=1)

    return df

# add ngram columns
df = add_top_ngrams(df, text_column='cleaned_text', top_k=10, ngram_range=(2, 2))
df = add_top_ngrams(df, text_column='cleaned_text', top_k=10, ngram_range=(3, 3))

# average review length for each sentiment class
print("\nAverage review length by sentiment:")
print(df.groupby('sentiment_binary')['review_length'].mean())

# most common words from text series
def get_top_words(text_series):
    words = ' '.join(text_series).split()
    return Counter(words).most_common(10)

# top words for positive and negative reviews
top_pos = get_top_words(df[df['sentiment_binary'] == 1]['cleaned_text'])
top_neg = get_top_words(df[df['sentiment_binary'] == 0]['cleaned_text'])

# convert top words to dfs 
pos_df = pd.DataFrame(top_pos, columns=['Word', 'Count'])
neg_df = pd.DataFrame(top_neg, columns=['Word', 'Count'])

# plot the top 10 words in positive and negative reviews
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(data=pos_df, x='Count', y='Word', color='green')
plt.title("Top Positive Words")

plt.subplot(1, 2, 2)
sns.barplot(data=neg_df, x='Count', y='Word', color='red')
plt.title("Top Negative Words")

plt.tight_layout()
plt.show()

# features to use for training
training_features = [
    'has_upvotes',
    'review_length',
    'exclamation_count',
    'question_count'
] + [col for col in df.columns if col.startswith('ngram_')]

# new df of selected features
new_cols = ['cleaned_text', 'sentiment_binary'] + training_features
df_selected = df[new_cols]

# save new df for training
df_selected.to_csv("data/processed/preprocessed_reviews.csv", index=False)
print("Selected features saved to: data/processed/preprocessed_reviews.csv")
