import os
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle
import json
nltk.download('wordnet')
nltk.download('omw-1.4')

data = pd.read_csv('Topic/data job posts.csv')

df = data[data['IT']]
# selecting 
cols = ['RequiredQual', 'Eligibility', 'Title', 'JobDescription', 'JobRequirement']
df = df[cols]

classes = df['Title'].value_counts()[:21]
keys = classes.keys().to_list()

df = df[df['Title'].isin(keys)]
df['Title'].value_counts()


def change_titles(x):
    x = x.strip()
    if x == 'Senior Java Developer':
        return 'Java Developer'
    elif x == 'Senior Software Engineer':
        return 'Software Engineer'
    elif x == 'Senior QA Engineer':
        return 'Software QA Engineer'
    elif x == 'Senior Software Developer':
        return 'Senior Web Developer'
    elif x == 'Senior PHP Developer':
        return 'PHP Developer'
    elif x == 'Senior .NET Developer':
        return '.NET Developer'
    elif x == 'Senior Web Developer':
        return 'Web Developer'
    elif x == 'Database Administrator':
        return 'Database Admin/Dev'
    elif x == 'Database Developer':
        return 'Database Admin/Dev'

    else:
        return x


df['Title'] = df['Title'].apply(change_titles)
df['Title'].value_counts()


class LemmaTokenizer(object):
    def __init__(self):
        # lemmatize text - convert to base form 
        self.wnl = WordNetLemmatizer()
        # creating stopwords list, to ignore lemmatizing stopwords 
        self.stopwords = stopwords.words('english')

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.stopwords]


# # train features and labels
y = df['Title']
X = df['RequiredQual']
# tfidf feature rep
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
vectorizer.fit(X)
pickle.dump(vectorizer, open('Topic/vectorizer.sav', 'wb'))
# transforming text to tfidf features
tfidf_matrix = vectorizer.transform(X)
# sparse matrix to dense matrix for training
X_tfidf = tfidf_matrix.toarray()
# encoding text labels in categories 
enc = LabelEncoder()
enc.fit(y.values)
y_enc = enc.transform(y.values)


current_vector = pickle.load(open("Topic/vectorizer.sav", "rb"))
clf_classifier: object = pickle.load(open("Topic/Logistic_Model.sav", "rb"))


def User_Input(dict_data):
    vec = current_vector.transform(dict_data)
    result = clf_classifier.predict(vec)
    required_post = enc.inverse_transform(result)
    result = list(result)
    required_post = required_post[result.index(max(result))]
    return_result = required_post
    os.remove("experience.json")
    return return_result




