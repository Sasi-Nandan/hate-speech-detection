
import os
import sklearn
import numpy as np
import pandas as pd
import csv
from pandas import DataFrame
import string
import time

os.chdir(os.path.dirname(os.getcwd()))
parentDir = os.getcwd()
os.chdir(os.getcwd()+"/data")
predictionsDir = parentDir+"/predictions"
dataDir = parentDir+"/data"
task1Dir = parentDir+"/task1"

def show_hateful_count( file_name ):
    train_set = csv.reader( open(file_name), delimiter=",", quoting=csv.QUOTE_NONE )
    count_print = 0
    for row in train_set:
        if ( row[1] == "1" ):
            count_print += 1        
    print(file_name, end=" ")
    print(count_print)

def create_predictions( predictions, test_df, file_name ):
    row_list = []
    row_list.append(["id","hateful"])
    for i in range(len(test_df)):
        curr = []
        curr.append(str(test_df.iloc[i][0]))
        curr.append(predictions[i])
        row_list.append(curr)
    os.chdir(predictionsDir)
    with open(file_name, 'w', newline='') as file:
        csv.writer(file).writerows(row_list)

##### data preprocessing
start = time.time()
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

print("Preprocessing done in:", end=" ")
print(time.time() - start)

train_df = pd.DataFrame( train_df[1:], columns = ['id', 'text', 'hateful'] )
test_df = pd.DataFrame( test_df[1:], columns = ['id', 'text'] )

start = time.time()
########## Random Forest classifier using TF IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 5)
train_word_corpus = train_df.iloc[:,1].tolist()
test_word_corpus = test_df.iloc[:,1].tolist()
train_x = vectorizer.fit_transform(train_word_corpus).toarray()
train_y = train_df.iloc[:,2].tolist()
valid_x = vectorizer.transform(test_word_corpus)#.toarray()
clf = RandomForestClassifier()
clf.fit(train_x, train_y)
RF_valid_output = clf.predict(valid_x)
create_predictions( RF_valid_output, test_df, file_name = "RF.csv" )

print("Random Forest done in:", end=" ")
print(time.time() - start)

start = time.time()
######### svm classifier
import spacy
from sklearn import svm

nlp = spacy.load('en_core_web_md')
svm_train_x = []
for tweet in train_df.iloc[:,1]:
    tweet_word_embd = nlp(tweet)
    svm_train_x.append(tweet_word_embd.vector)

svm_train_y = train_df.iloc[:,2]
clf = svm.SVC()
clf.fit(svm_train_x, svm_train_y)
svm_valid_x = []
for tweet in test_df.iloc[:,1]:
    tweet_word_embd = nlp(tweet)
    svm_valid_x.append(tweet_word_embd.vector)
SVM_valid_output = clf.predict(svm_valid_x)
create_predictions( SVM_valid_output, test_df, file_name="SVM.csv" )

print("SVM done in:", end=" ")
print(time.time() - start)

start = time.time()
##### fasttext classifier
import fasttext

f = open("fasttext_train.txt", "a")
for index, row in train_df.iterrows():
    f.write("__label__" + row[2] + " " + row[1] + "\n")
f.close()
model = fasttext.train_supervised('fasttext_train.txt')
valid_x = []
for tweet in test_df.iloc[:,1]:
    valid_x.append(tweet)
os.remove("fasttext_train.txt")
output_tuples = model.predict( valid_x )
FT_valid_output = []
for i in range(test_df.shape[0]):
    FT_valid_output.append(output_tuples[0][i][0][9:])

create_predictions( FT_valid_output, test_df, file_name="FT.csv" )

print("Fasttext done in:", end=" ")
print(time.time()-start)
print("Done!!")