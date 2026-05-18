import streamlit as st
import pickle
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# LOAD MODEL
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stemmer = PorterStemmer()

# CLEAN FUNCTION
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

# TITLE
st.title("NLP SENTIMENT ANALYSIS SYSTEM")

# OPTION MENU
option = st.selectbox(
    "Choose Option",
    ("Input Text", "Upload File")
)

# INPUT TEXT OPTION
if option == "Input Text":

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

# FILE UPLOAD OPTION
elif option == "Upload File":

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.write(data.head())

        if 'review' in data.columns:

            clean_reviews = data['review'].apply(clean_text)

            vectors = tfidf.transform(clean_reviews).toarray()

            predictions = model.predict(vectors)

            data['Prediction'] = predictions

            st.write(data)

        else:
            st.error("CSV must contain 'review' column")
