import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
# from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


from datetime import datetime, timedelta

import time
import os
import random
# from document import Document
from collections import defaultdict, Counter

t_path = "datasets/"

all_docs = defaultdict(lambda: list())

topic_list = list()
text_list = list()

print("Reading all the documents...\n")

for topic in os.listdir(t_path):
    d_path = t_path + topic + '/'

    for f in os.listdir(d_path):
        f_path = d_path + f
        file = open(f_path,'r',encoding='unicode_escape')
        text_list.append(file.read())
        file.close()
        topic_list.append(topic)

title_train, title_test, category_train, category_test = train_test_split(text_list,topic_list, test_size=0.2)
title_train, title_dev, category_train, category_dev = train_test_split(title_train,category_train, test_size=0.2)

print("Training: ",len(title_train))
print("Developement: ",len(title_dev),)
print("Testing: ",len(title_test))

#tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
import string
from nltk.stem import WordNetLemmatizer
lementizer = WordNetLemmatizer()

def tokenizer(sentence):
    no_punct =""
    for word in sentence:
        if word not in string.punctuation:
            no_punct = no_punct + word
    
    str_li = list(no_punct.split(" "))    
    
    lem_txt = [lementizer.lemmatize(word) for word in str_li]
    no_stop = [word for word in lem_txt if word not in stop_words]
    return no_stop

stop_words = nltk.corpus.stopwords.words("english")
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words)
vectorizer.fit(iter(title_train))
Xtr = vectorizer.transform(iter(title_train))
Xde = vectorizer.transform(iter(title_dev))
Xte = vectorizer.transform(iter(title_test))



encoder = LabelEncoder()
encoder.fit(category_train)
Ytr = encoder.transform(category_train)

Yde = encoder.transform(category_dev)
Yte = encoder.transform(category_test)

print(encoder.classes_)