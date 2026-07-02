import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")

print(df.head())

# Clean Dataset
df = df[['v1','v2']]

df.columns = ['label','message']

df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})

#Text Preprocessing
import nltk

nltk.download('stopwords')
nltk.download('punkt')

import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

ps = PorterStemmer()

def clean_text(text):

    text = text.lower()

    text = re.sub("[^a-zA-Z]", " ", text)

    words = text.split()

    words = [ps.stem(word) for word in words if word not in stopwords.words("english")]

    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

#Train Test Split
from sklearn.model_selection import train_test_split

X = df["message"]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(

X,

y,

test_size=0.2,

random_state=42
)

#Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

#Train Model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train,y_train)

#Prediction
predictions = model.predict(X_test)

#Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,predictions)

print(accuracy)

#Confusion Matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predictions))

#Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

#Save Model
import pickle

pickle.dump(model,open("model/spam_model.pkl","wb"))

pickle.dump(vectorizer,open("model/vectorizer.pkl","wb"))