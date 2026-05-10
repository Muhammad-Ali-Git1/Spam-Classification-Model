import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()
def transform_text(text):

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)
st.title("📧 Email/SMS Spam Classifier")
st.write("Enter a message and check if it's spam or not")

input_sms = st.text_area("Enter your message here:")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error("This is SPAM message")
    else:
        st.success("This is NOT spam (Ham)")