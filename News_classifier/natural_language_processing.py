# Natural Language Processing

import pandas as pd
import codecs
import re

# Importing the dataset
dataset = pd.read_csv('News_classifier/all_news.tsv', delimiter = '\t', quoting = 3)


with codecs.open('News_classifier/excluded_word_list_out.txt','r',encoding='utf8') as f:
                text = f.read()
                #print(len(text))

excluded = set(text.split())



corpus = []


def clean_data(text):
    cleaned = text
    cleaned = re.sub(r'\[', ' ', cleaned)
    cleaned = re.sub(r'\]', ' ', cleaned)
    cleaned = re.sub(r'\sও\sও\s', ' ', cleaned)
    cleaned = re.sub(
        r'\sÅ\s|\sটি\s|\sঅ\s|\sআ\s|\sঈ\s|\sউ\s|\sঊ\s|\sঋ\s|\sঔ\s|\sক\s|\sখ\s|\sগ\s|\sঘ\s|\sঙ\s|\sচ\s|\sছ\s|\sজ\s|\sঝ\s|\sঞ\s|\sট\s|\sঠ\s|\sড\s|\sঢ\s|\sণ\s|\sত\s|\sথ\s|\sদ\s|\sধ\s|\sন\s|\sপ\s|\sফ\s|\sব\s|\sভ\s|\sম\s|\sয\s|\sর\s|\sল\s|\sশ\s|\sষ\s|\sস\s|\sহ\s|\sড়\s|\sঢ়\s|\sয়\s',
        ' ', cleaned)
    cleaned = re.sub(r'[,:\'\"?।১২৩৪৫৬৭৮৯.!@#$%^–&*()_+;\-০‘’/=—╛×╛…﻿¡>|`}¤¦¯©¹ºÑøÿ]', ' ', cleaned)
    cleaned = re.sub(r'[a-zA-Z0-9]',' ',cleaned)
    return cleaned

for i in range(0, 769):

    review = dataset['News'][i]
    review = clean_data(review)
    review = review.split()

    review = [word for word in review if not word in excluded]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()


y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#classifier = GaussianNB()
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)


def predict_news(news_input):
    cleaned = clean_data(news_input)
    news_list = []
    news_list.append(cleaned)
    news = cv.transform(news_list).toarray()

    news_pred = classifier.predict(news)
    return news_pred



