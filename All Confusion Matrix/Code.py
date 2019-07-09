# Importing the libraries
import os
import codecs
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import codecs
from sklearn.metrics import classification_report, confusion_matrix 


#Preprocess Data
start = time.time()
with codecs.open('8International.txt','r',encoding='utf8') as f:
                text = f.read()
                print(len(text))
countRe = re.compile(r'\t')
print("no of tab before : "+str(len(countRe.findall(text))))

single_lined = re.sub(r'\s+', ' ', text)

countRe = re.compile(r'\t')
print("no of tab: "+str(len(countRe.findall(single_lined))))
tagged = re.sub(r'</news>\s+', '\t8\n', single_lined)

countRe = re.compile(r'\n')
print("no of newline: "+str(len(countRe.findall(tagged))))

countRe = re.compile(r'\t')
print("no of tab: "+str(len(countRe.findall(tagged))))

cleaned = re.sub(r'<date>|</date>|<title>|</title>|<news>','',tagged)

with codecs.open('8International_out.tsv','w',encoding='utf8') as f:
    f.write(cleaned)

print("total time : "+str(time.time()-start))


#Join all tsv files from news directory
start = time.time()
output = str()
for root, dirs, files in os.walk("news"):
   for file in files:
       if file.endswith(".tsv"):
            file_name=os.path.join(root, file)
            with codecs.open(file_name,'r',encoding='utf8') as f:
               text = f.read()
               print(len(text))
               output+=text
with codecs.open('Dataset.tsv','w',encoding='utf8') as f:
   f.write(output)
print("total time : "+str(time.time()-start))


# Importing the dataset
#dataset = pd.read_csv('news_out.tsv', delimiter = '\t', quoting = 3)
dataset = pd.read_csv('Dataset.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
with codecs.open('excluded_word_list_out.txt','r',encoding='utf8') as f:
                text = f.read()
                print(len(text))

excluded = set(text.split())


corpus = []
for i in range(0, 6530):
    review = dataset['News'][i]
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Classifier :  1
#---------------
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


#Classifier :  2
#---------------
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


error = []

# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')


#Classifier :  3
#---------------
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


#Classifier :  4
#---------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=8, n_estimators=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


#Classifier :  5
#---------------
from sklearn.svm import SVC
classifier = SVC(kernel="linear", C=0.025)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))