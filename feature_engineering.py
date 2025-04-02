import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import re

# TF-IDF vectorizer
def extract_tfidf_features(df_texts, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(
        max_features = max_features, 
        stop_words = 'english', 
        ngram_range = (1, 2)
    )
    return tfidf_vectorizer.fit_transform(df_texts), tfidf_vectorizer

# LDA topic distribution
def extract_lda_features(df_texts):
    count_vectorizer = CountVectorizer(
        max_features = 1000,  
        max_df = 0.95,        # filter out words that appear in more than 95% of documents
        min_df = 10,          # filter out words that appear in less than 10 documents
        stop_words = 'english'
    )
    X = count_vectorizer.fit_transform(df_texts)
    lda = LatentDirichletAllocation(
        n_components = 5, 
        random_state = 0,
        n_jobs = -1,           
        max_iter = 10,         
    )
    lda_topics = lda.fit_transform(X)
    return lda_topics, lda, count_vectorizer

# Sentiment polarity
def extract_sentiment_scores(df_texts):
    return np.array([[TextBlob(text).sentiment.polarity] for text in df_texts])

# Text statistics
def extract_text_statistics(df_texts):
    stats = []
    for text in df_texts:
        words = text.split()
        word_lengths = [len(word) for word in words]
        stats.append([
            len(text),                                 # num characters
            len(words),                                # num words
            np.mean(word_lengths) if word_lengths else 0,  # avg word length
            sum(1 for c in text if c.isupper()),       # uppercase count
            len(re.findall(r'[.!?]', text)),           # punctuation count
            len(re.findall(r"\bi\b|\bme\b|\bmy\b", text, re.IGNORECASE))  # 1st person pronouns
        ])
    return np.array(stats)

# Standardize dense features
def standardize_features(dense_features):
    scaler = StandardScaler()
    return scaler.fit_transform(dense_features), scaler