# This file opens "features.csv" and trains and tests multiple classifiers
# on the feature data. 
#


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from random import randrange
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

features = pd.read_csv("features.csv", index_col = False)


def SVM ():
# SVM classifier that works on all features
#
#
    x = features.drop(['label'], axis=1)

    y = features['label']


    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.precision_score(y_test, y_pred)
    precision = metrics.recall_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    
    return accuracy, recall, precision, f_score

def allClassifiers ():
# This method trains and tests multiple classifiers with different kernels
# The RBF classifier achieved the best results
#
    x = features.drop(['label'], axis=1)

    y = features['label']


    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)
    
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    accuracies = []
    f_scores = []
    for kernel in kernels:
        if kernel == "poly":
            model = SVC(kernel = kernel, degree = 2)
        else:
            model = SVC(kernel = kernel)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        f_scores.append(metrics.f1_score(y_test, y_pred))
    print()
    print("Testing all the features on all kernels")
    for index, kernel in enumerate(kernels):
        print(kernel, "kernel achieved the following results:", accuracies[index], "Accuracy", f_scores[index], "F-score")
        
    

def baseline ():
# This method trains and tests the baseline classifiers, DummyClassifiers,
# on all the features
#
    x = features.drop(['label'], axis=1)

    y = features['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)

    strats = ['most_frequent', 'stratified', 'uniform']
    acc = []
    rec = []
    pre = []
    for s in strats:
        clss = DummyClassifier(strategy = s)
        clss.fit(x_train, y_train)
        y_pred = clss.predict(x_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))
        rec.append(metrics.precision_score(y_test, y_pred))
        pre.append(metrics.recall_score(y_test, y_pred))
        
    
    mean_acc = round(sum(acc) / len(acc), 2)
    mean_rec = round(sum(rec) / len(rec), 2)
    mean_pre = round(sum(pre) / len(pre), 2)
    
    return mean_acc, mean_rec, mean_pre


def most_frequent ():
# This method trains and tests one of the baseline classifiers, the
# "most frequent" classifier. This classifier always predicts the most frequent
# label in the training set, disregarding all other features. 
    x = features.drop(['label'], axis=1)

    y = features['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)
    
    clss = DummyClassifier(strategy = "most_frequent")
    clss.fit(x_train, y_train)
    y_pred = clss.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    rec = metrics.precision_score(y_test, y_pred)
    pre = metrics.recall_score(y_test, y_pred)

    print("The baseline classifier got the following results:")
    print("Accuracy:", acc, "Recall:", rec, "Precision:", pre)
    

def individualTests ():
# This method trains and tests a classifier on all the features individually
#
#
    x_sentiment = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_ner = features.drop(['label', 'sentiment_score', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_length = features.drop(['label', 'ner', 'sentiment_score', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_exclmark = features.drop(['label', 'ner', 'length', 'sentiment_score', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_quesmark = features.drop(['label', 'ner', 'length', 'excl_marks', 'sentiment_score', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_contradiction = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'sentiment_score', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_temporal = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'sentiment_score', 'nouns', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_nouns = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'sentiment_score', 'adjectives', 'verbs', 'determiners', 'numbers'], axis=1)
    x_adjectives = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'sentiment_score', 'verbs', 'determiners', 'numbers'], axis=1)
    x_verbs = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'sentiment_score', 'determiners', 'numbers'], axis=1)
    x_determiners = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'sentiment_score', 'numbers'], axis=1)
    x_numbers = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'sentiment_score'], axis=1)

    x_random = features.drop(['label', 'ner', 'length', 'excl_marks', 'ques_marks', 'contradictions', 'temporal', 'nouns', 'adjectives', 'verbs', 'determiners', 'sentiment_score', 'numbers'], axis=1)
    x_random["random"] = len(features['label']) * [0]

    tests = [x_sentiment, x_ner, x_length, x_exclmark, x_quesmark, x_contradiction, x_temporal, x_nouns, x_adjectives, x_verbs, x_determiners, x_numbers, x_random]
    names = ["Sentiment", "Names Entity Recognition", "Length", "Exclamation mark", "Question mark", "Contradiction", "Temporal", "Noun", "Adjective", "Verb", "Determiner", "Number", "Random"]
    y = features['label']
    accuracies = []
    f_scores = []
    for test in tests:
        x_train, x_test, y_train, y_test = train_test_split(test, y, train_size = 0.75, test_size = 0.25)

        model = SVC()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        f_scores.append(metrics.f1_score(y_test, y_pred))
        
    print("Accuracies:")
    for index,i in enumerate(accuracies):
        print(names[index], "=",i, end=", ")
    print()

    print("F-scores:")
    for index,i in enumerate(f_scores):
        print(names[index], "=",i, end=", ")
    print()
        
        
    
def bestPerforming ():
# This method trains and tests a classifier using only the 4 features with the 
# best individual performance. The used features: Length feature, Question mark
# feature, Nouns feature and Determiners feature. 
    x = features.drop(['label', 'ner', 'contradictions', 'temporal', 'adjectives', 'numbers'], axis=1)
    
    y = features['label']


    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.precision_score(y_test, y_pred)
    precision = metrics.recall_score(y_test, y_pred)
    f_score = metrics.f1_score(y_test, y_pred)
    
    return accuracy, recall, precision, f_score
    


def plotje ():
# This method plots, from both the satirical and non-satirical data,
# the length feature on the x-axis and the nouns feature on the y-axis
#

    satire = features[features["label"] == 1]
    notsatire = features[features["label"] == 0]

    xs = satire["length"][:40]
    xn = notsatire["length"][:40]

    ys = satire["nouns"][:40]
    yn = notsatire["nouns"][:40]
    
    plt.figure(figsize=(20,8))
    sat = plt.scatter(xs, ys, marker = '+', color='blue')
    nsat = plt.scatter(xn, yn, marker = '_', color='red')
    plt.xlabel("Length")
    plt.ylabel("Nouns")
    plt.legend((sat,nsat), ("Satire", "Non-satire"))
    plt.show()




#plotje()
accuracy, recall, precision, f_score = SVM()

accuracyt, recallt, precisiont = baseline()

individualTests()

print("Results of the baseline classifiers:")
print("Mean accuracy: ", accuracyt, "Mean recall: ", recallt, "Mean precision: ", precisiont)


#most_frequent()

print("Your classifier achieved the following results:")
print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
print("F-score: ", f_score)


bestaccuracy, bestrecall, bestprecision, bestf_score = bestPerforming()
print()
print("Results of a classifier with only the best performing features:")
print("Accuracy: ", bestaccuracy)
print("Recall: ", bestrecall)
print("Precision: ", bestprecision)
print("F-score: ", bestf_score)

allClassifiers()





