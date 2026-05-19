import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import sys

print("Loading dataset...")
news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

print("Downloading stopwords...")
nltk.download('stopwords', quiet=True)
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

print("Applying stemming... (this might take a minute)")
news_dataset['content'] = news_dataset['content'].apply(stemming)

X = news_dataset['content'].values
Y = news_dataset['label'].values

print("Fitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

print("Training LogisticRegression model...")
# Using LogisticRegression since it was the one evaluated in the notebook (98% accuracy) and trains very quickly.
model = LogisticRegression()
model.fit(X_vectorized, Y)

print("Saving model and vectorizer...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training complete! Saved model.pkl and vectorizer.pkl.")
