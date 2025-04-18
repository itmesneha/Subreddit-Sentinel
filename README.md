# Subreddit-Sentinel
CS5246 TEXT MINING PROJECT

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

