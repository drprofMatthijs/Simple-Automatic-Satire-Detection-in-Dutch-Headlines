# This file calculates some statistics from the headlines database 
#
#

import spacy
import string
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from random import randrange

#features = pd.read_csv("features.csv", index_col = False)

headlines = pd.read_csv("headlines_dataset.csv", index_col = False)

labels = headlines['is_sarcastic']

satire = headlines[headlines['is_sarcastic'] == 1]

not_satire = headlines[headlines['is_sarcastic'] == 0]


sperc = round((len(satire) / len(labels)) * 100, 2)
nperc = round((len(not_satire) / len(labels)) * 100, 2)


satire_politics = satire[satire['is_politics'] == 1]
satire_foreign = satire[satire['is_foreign'] == 1]
satire_domestic = satire[satire['is_domestic'] == 1]

notsatire_politics = not_satire[not_satire['is_politics'] == 1]
notsatire_foreign = not_satire[not_satire['is_foreign'] == 1]
notsatire_domestic = not_satire[not_satire['is_domestic'] == 1]



print("The database has the following statistics:")
print("Number of headlines in the database =", len(labels))
print("Of these headlines,", len(satire), "are satire and", len(not_satire), "are not satire")
print("In percentages:", sperc, "% satire &", nperc, "% not satire")
print("The satire headlines contain", len(satire_politics), "political headlines,", len(satire_foreign), "foreign headlines &", len(satire_domestic), "domestic headlines")
print("In precentages:", round((len(satire_politics) / len(satire)) * 100, 2), "% politics,", round((len(satire_foreign) / len(satire)) * 100, 2), "% foreign &", round((len(satire_domestic) / len(satire)) * 100, 2), "% domestic")
print("The not-satire headlines contain", len(notsatire_politics), "political headlines,", len(notsatire_foreign), "foreign headlines &", len(notsatire_domestic), "domestic headlines")
print("In precentages:", round((len(notsatire_politics) / len(not_satire)) * 100, 2), "% politics,", round((len(notsatire_foreign) / len(not_satire)) * 100, 2), "% foreign &", round((len(notsatire_domestic) / len(not_satire)) * 100, 2), "% domestic")
print()
print("Average length of satire vs not-satire:")
len_satire = []
len_notsatire = []

satire_headlines = satire["headline"]
notsatire_headlines = not_satire["headline"]
for i in satire_headlines:
    len_satire.append(len(i))

for i in notsatire_headlines:
    len_notsatire.append(len(i))

lengthsat = round(sum(len_satire) / len(len_satire), 3)
lengthnotsat = round(sum(len_notsatire) / len(len_notsatire), 3)
print("Satire =", lengthsat)
print("Not satire =", lengthnotsat)












