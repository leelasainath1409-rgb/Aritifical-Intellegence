import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stemmer = PorterStemmer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

st.title("NLP Sentiment Analysis")

review = st.text_area("Enter Review")

if st.button("Predict"):

    cleaned_review = clean_text(review)

    vector_input = tfidf.transform([cleaned_review]).toarray()

    prediction = model.predict(vector_input)

    if prediction[0] == 2:
        st.success("Positive Sentiment")

    elif prediction[0] == 1:
        st.warning("Neutral Sentiment")

    else:
        st.error("Negative Sentiment")
