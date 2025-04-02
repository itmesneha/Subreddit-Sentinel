from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
tweetTokenizer = TweetTokenizer()

def cleanText(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text) # Remove Reddit formatting tags [text], (text)
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters and punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers

    return text

def tokenize(text):
    #resultBert = tokenizer.tokenize(text) #BERTokenizer
    #resultNLTK = word_tokenize(text) #NLTK
    resulttweetNLTK = tweetTokenizer.tokenize(text) #NLTKTweet
    return resulttweetNLTK

def lemmatize(tokens):
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens 
                    if word not in stop_words and len(word) > 1]
    return cleaned_tokens

def limit_subreddits(df):
    top_5 = df['subreddit'].value_counts().head(5).index
    df_processed = df[df['subreddit'].isin(top_5)]
    return df_processed

def preprocess(df):
    df_processed = df.copy()
    df_processed.dropna(inplace=True)
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    df_processed.drop(columns=['created_utc'], inplace=True)
    return df_processed

def text_process(text):
    text = cleanText(text) # Clean
    tokens = tokenize(text) # Tokenize
    #tokens = lemmatize(tokens)

    return ' '.join(tokens)

