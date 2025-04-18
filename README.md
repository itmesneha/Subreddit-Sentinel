# Subreddit Sentinel: Early Mental Health Detection from Reddit Posts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

We propose Subreddit Sentinel, an end-to-end system
for early detection of mental health concerns by mining Reddit
posts. Leveraging a novel semantic–temporal sampling pipeline
on over 1.8 million posts from five mental-health subreddits,
we construct a multi-level feature suite comprising sentiment
(VADER, TextBlob), emotion categories (Empath), latent top-
ics (LDA), and TF–IDF. We compare a battery of classi-
fiers—including logistic regression, Random Forests, LightGBM,
voting and stacking ensembles—against fine-tuned domain-
adapted transformers (Mental-BERT, Mental-RoBERTa). Our
stacking ensemble achieves the best macro-F1 of 0.78, while
Mental-RoBERTa attains an F1 of 0.75. We demonstrate that
semantic-temporal sampling preserves discourse diversity and
that ensemble and transformer methods complement each other
in accuracy and explainability. 

## Key Features

- **Novel Semantic-Temporal Sampling**: Preserves discourse diversity across 1.8 million Reddit posts
- **Multi-Level Feature Engineering**: Combines sentiment analysis, emotion categories, latent topics, and TF-IDF vectors
- **Comprehensive Model Comparison**: Evaluates traditional ML, ensemble methods, and transformer approaches
- **Domain-Specific Transformers**: Leverages Mental-BERT and Mental-RoBERTa pre-trained on mental health corpora
- **High Classification Performance**: Achieves 0.78 macro-F1 with stacking ensemble and 0.75 F1 with Mental-RoBERTa

## Data

We use the Reddit Mental Health Dataset (RMHD) containing 1,851,580 posts from five mental health subreddits:
- r/anxiety
- r/depression
- r/mentalhealth
- r/suicidewatch
- r/lonely

## Preprocessing Pipeline

1. Reddit-specific noise removal
2. Text standardization while preserving psychological signals
3. Custom stopword filtering (preserving clinical terms)
4. Lemmatization for term normalization
5. Semantic-temporal sampling for balanced representation

## Feature Engineering

- **Sentiment Analysis**: VADER and TextBlob for polarity measurement
- **Emotion Categories**: Empath lexicon for fine-grained emotion detection
- **Topic Modeling**: LDA to extract latent themes in discussions
- **Word Frequency**: TF-IDF scores to identify distinctive vocabulary
- **Distress Score**: Composite measure combining emotion and sentiment features
- **Temporal Analysis**: Tracking trends across time, including COVID-19 impact

## Models Evaluated

### Traditional ML Models
- Logistic Regression
- Support Vector Machines
- Neural Networks (MLP)
- Random Forest

### Ensemble Methods
- Bagging with SVM
- Voting Classifier
- Stacking Classifier

### Boosting Models
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### Transformer Models
- Mental-BERT (domain-adapted)
- Mental-RoBERTa (domain-adapted)

## Key Results

- Domain-adapted transformers outperform traditional ML approaches
- Mental-RoBERTa: F1 score of 0.715, accuracy of 0.723
- Stacking Classifier: F1 score of 0.581, accuracy of 0.585
- High performance on anxiety detection (F1 > 0.87)
- Challenges in distinguishing depression from suicidal ideation

## Limitations and Future Work

- Addressing contextual ambiguity between related conditions
- Incorporating temporal dynamics of user posting history
- Bridging the gap between transformers and feature-engineered approaches
- Enhancing model explainability for clinical applications
- Validating across different mental health communities
- Addressing ethical considerations around privacy and intervention

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/subreddit-sentinel.git
cd subreddit-sentinel

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



