# Spam Email Detection App

A simple and interactive web application built with Streamlit that detects whether an email (SMS message) is spam or not using Natural Language Processing and machine learning.


PROJECT OVERVIEW

This project uses a labeled dataset of SMS messages to train a classification model that distinguishes between **spam** and **ham (not spam)**. The model is deployed using Streamlit, allowing users to input a message and get instant predictions.

FEATURES

- Clean and preprocess email text data
- Transform text using TF-IDF vectorization
- Train and evaluate a Multinomial Naive Bayes classifier
- Streamlit interface for message input and prediction
- Fast, lightweight, and easy to use

TECHNOLOGIES USED

- **Python**
- **Pandas** – Data handling
- **Scikit-learn** – Machine learning pipeline (TF-IDF, Naive Bayes, train-test split)
- **Streamlit** – Web application framework
- **NLP** – Text cleaning and vectorization



DATASET
The app uses the **SMS Spam Collection Dataset**, which includes over 5,000 SMS messages labeled as spam or ham. It can be found here: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)




