# This file trains and tests the classifier on the different kinds of news seperately.
# Political news, domestic news and foreign news.
#

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from random import randrange
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

features = pd.read_csv("features.csv", index_col = False)
headl = pd.read_csv("headlines_dataset.csv")

features["is_politics"] = headl["is_politics"]
features["is_domestic"] = headl["is_domestic"]
features["is_foreign"] = headl["is_foreign"]


features_pol = features[features["is_politics"] == 1]
features_dom = features[features["is_domestic"] == 1]
features_for = features[features["is_foreign"] == 1]


def is_politics ():
# SVM classifier that works on all features
# only using foreign headlines
#
    x = features_pol.drop(['label', 'is_politics', 'is_domestic', 'is_foreign'], axis=1)

    y = features_pol['label']


    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.precision_score(y_test, y_pred)
    precision = metrics.recall_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    
    print("Testing the classifier on political headlines:")
    print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision, "F_score:", f_score)


def is_domestic ():
# SVM classifier that works on all features
# only using domestic headlines
#
    x = features_dom.drop(['label', 'is_politics', 'is_domestic', 'is_foreign'], axis=1)

    y = features_dom['label']


    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.precision_score(y_test, y_pred)
    precision = metrics.recall_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    
    print("Testing the classifier on domestic headlines:")
    print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision, "F_score:", f_score)

def is_foreign ():
# SVM classifier that works on all features
# only using political headlines
#
    x = features_for.drop(['label', 'is_politics', 'is_domestic', 'is_foreign'], axis=1)

    y = features_for['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.precision_score(y_test, y_pred)
    precision = metrics.recall_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    
    print("Testing the classifier on foreign headlines:")
    print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision, "F_score:", f_score)



def main():
    is_politics()
    print()
    is_domestic()
    print()
    is_foreign()


main()



    
    
