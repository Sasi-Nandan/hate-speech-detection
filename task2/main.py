import os
import sklearn
import numpy as np
import pandas as pd
import csv
from pandas import DataFrame
import string
import time
import spacy
import sys 

start = time.time()
os.chdir(os.path.dirname(os.getcwd()))
parentDir = os.getcwd()
dataDir = os.getcwd()+"/data"
predictionDir = os.getcwd()+"/predictions"
os.chdir(dataDir)

read_tsv = open("train.tsv")
train_set = csv.reader( read_tsv, delimiter="\t", quoting=csv.QUOTE_NONE )
train_df = []
for row in train_set:
    curr = []
    curr.append(row[0].translate(str.maketrans('', '', string.punctuation)).lower())
    curr.append(row[1].translate(str.maketrans('', '', string.punctuation)).lower())
    curr.append(row[2].translate(str.maketrans('', '', string.punctuation)).lower())
    train_df.append(curr)
read_tsv.close()

read_tsv = open("test.tsv")
test_set = csv.reader( read_tsv, delimiter="\t", quoting=csv.QUOTE_NONE )
test_df = []
for row in test_set:
    curr = []
    curr.append(row[0].translate(str.maketrans('', '', string.punctuation)).lower())
    curr.append(row[1].translate(str.maketrans('', '', string.punctuation)).lower())
    test_df.append(curr)
read_tsv.close()

train_df = pd.DataFrame(train_df[1:], columns = ['id', 'text', 'hateful'])
test_df = pd.DataFrame( test_df[1:], columns = ['id', 'text'] )

nlp = spacy.load('en_core_web_md')
x_train = []
for tweet in train_df.iloc[:,1]:
    lemmatsd_tweet = " ".join([token.lemma_ for token in nlp(tweet)])
    x_train.append(nlp(lemmatsd_tweet).vector)    

y_train = train_df.iloc[:,2]

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=1000)
clf.fit(np.asarray(x_train), np.asarray(y_train))

x_test = []
for tweet in test_df.iloc[:,1]:
    lemmatsd_tweet = " ".join([token.lemma_ for token in nlp(tweet)])
    x_test.append(nlp(lemmatsd_tweet).vector)

task2_valid_output = clf.predict(np.asarray(x_test))

row_list = []
row_list.append(["id","hateful"])
for i in range(len(test_df)):
    curr = []
    curr.append(str(test_df.iloc[i][0]))
    curr.append(task2_valid_output[i])
    row_list.append(curr)

os.chdir(predictionDir)
with open('T2.csv', 'w', newline='') as file:
    csv.writer(file).writerows(row_list)

print("Time taken: ", end=" ")
print(time.time() - start)
sys.exit("Done!!")
