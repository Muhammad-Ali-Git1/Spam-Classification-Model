
import pandas as pd
import numpy as np
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score


nltk.download('punkt')
nltk.download('stopwords')

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

df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates(keep='first')

df['transformed_text'] = df['text'].apply(transform_text)

tf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X = tf.fit_transform(df['transformed_text']).toarray()

y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

model = MultinomialNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred))


pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tf, open('vectorizer.pkl', 'wb'))

print("\nModel and Vectorizer saved successfully!")
