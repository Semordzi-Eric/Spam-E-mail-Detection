import streamlit as st
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/DELL/Desktop/python/Classification/E-mail_spam_project/Data/spam.csv", encoding='latin1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# App
def main():
    st.title("Email Spam Detection App")

    df = load_data()
    st.write("Sample Data:")
    st.dataframe(df.sample(5))

    # Preprocessing
    df['clean_message'] = df['message'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['clean_message'])
    y = df['label']

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    st.write(f"Model Accuracy: **{acc:.2%}**")
    
    
    

    # Sidebar input
    st.sidebar.header("Enter an email message")
    user_input = st.sidebar.text_area("Message", height=200)

    if st.sidebar.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter a message to classify.")
        else:
            clean_input = clean_text(user_input)
            vector_input = vectorizer.transform([clean_input])
            prediction = model.predict(vector_input)[0]
            label = "SPAM" if prediction == 1 else "NOT SPAM"
            st.subheader("Prediction Result:")
            st.success(label)

if __name__ == "__main__":
    main()
