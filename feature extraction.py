# This file extracts all the features from the headlines database, 
# creating a new file "features.csv"
#

import pandas as pd
import time
import string
import spacy
from collections import Counter

class features:
    def __init__(self):
        headl = pd.read_csv("headlines_dataset.csv")
        print("The database is traversed two times, this will take several minutes")
        
        self.nlp = spacy.load('nl_core_news_sm')
        self.headlines = headl["headline"]
        self.headline_tokens, self.ner_feature, self.exclmark_feature, self.quesmark_feature, self.punctuation_feature, self.nouns_feature, self.adjectives_feature, self.number_feature, self.det_feature, self.verb_feature = self.iterateData(self.headlines) # features calculated with the unprocessed data

        

        self.sentiment_feature, self.length_feature, self.contra_feature, self.temp_feature = self.iterateTokens() # features calculated on the processed (lemmatized and tokenized) data
        #self.domestic = headl["is_domestic"]
        #self.foreign = headl["is_foreign"]
        #self.politics = headl["is_politics"]
        self.label = headl["is_sarcastic"]


    def iterateData (self, database):
    # This method propagates through the data to create a new list,
    # running the "tokenize" function on every string. 
    # At the same time, most of the features are calculated
    #
    
        lemmatized = []
        ner = []
        excl_mark = []
        ques_mark = []
        punctuation = []
        nouns = []
        adjectives = []
        numbers = []
        determ = []
        verbss = []
        
        leng = len(database)

        track = round(leng/20)
        adder = track
        percent = 5

        print("Starting to tokenize and lemmatize", leng,  "sentences, simultaneously calculating most of the features")

        for index,line in enumerate(database):
            if index == track:
                print(percent,"% done...", end="\r")
                track += adder
                percent += 5
            lemmatized.append(self.tokenize(line))                      #call tokenize function
            ner.append(self.namedEntities(line))                        #call NER runction
            excl_mark.append(line.count("!"))                           #count exclamation marks
            ques_mark.append(line.count("?"))                           #count question marks
            noun, adj, num, dets, verbs = self.pos_features(line)       #POS features
            nouns.append(noun)
            adjectives.append(adj)
            numbers.append(num)
            determ.append(dets)
            verbss.append(verbs)
                                                    #count all punctuation!!!!!!!!!!1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        print("First feature extraction part done")

        return lemmatized, ner, excl_mark, ques_mark, punctuation, nouns, adjectives, numbers, determ, verbss
        
    

    def iterateTokens(self):
    # This method traverses through the new list with the processed data, 
    # creating the last features. 
    #
        valences = pd.read_csv("word_valence_dutch_words.csv") 
        leng = len(self.headlines)
        initial = [0] * leng
        leng_feature = []
        contra_feature = []
        temporal_feature = []
        dutch_words = valences["Words"]
        dutch_valences = valences["Valence"]
        contradictives = open("signaalwoorden_tegenstelling.txt")
        contra = contradictives.read().splitlines()
        temporal = open("temporele_tegenstelling.txt")
        tempor = temporal.read().splitlines()

        print("Starting to calculate the last features for all sentences")
        
        track = round(leng/20)
        adder = track
        percent = 5
        
        
        for index,lines in enumerate(self.headline_tokens): 
            if index == track:
                print(percent,"% done...", end="\r")
                track += adder
                percent += 5

            
            begin = []
            con = 0
            tem = 0
            for words in lines:
                for i,val in enumerate(dutch_words):
                    if val == words:
                        begin.append(dutch_valences[i]) # sentiment feature
                for item in contra:
                    if item == words:
                        con += 1                        # contradiction feature
                for temp in tempor:
                    if temp == words:                   # temporal feature
                        tem += 1
                        
                        
                        
            sumlis = sum(begin)
            if sumlis == 0:
                initial[index] = 0              # sentiment feature
            else:
                final = sumlis / (len(begin))
                answer = round(final, 2)
                initial[index] = answer         # sentiment feature
                
                        
            contra_feature.append(con)          # contradiction feature

            temporal_feature.append(tem)        # temporal feature
                      
            leng_feature.append(len(lines))     # length feature

            
            
        contradictives.close()
        temporal.close()
        
        sumsent = sum(initial)
        lengsent = len(initial)
        meansent = round(sumsent / lengsent, 2)

        sentiment = [i if i != 0 else meansent for i in initial]
        print("Feature extraction done")

        return sentiment, leng_feature, contra_feature, temporal_feature
    

    def tokenize (self, sentence):
    # This function returns a list of the lemmatized tokens of a single sentence, removing punctuation and changing uppercase letters in lowercase.
    #
    #
    
        remove = sentence.translate(str.maketrans('','',string.punctuation)) #remove punctuation
        temp = remove.lower() #change all uppercase letters to lowercase
        
        sent = self.nlp(temp) 

        returner = []

        for token in sent:
            returner.append(token.lemma_)
            

        return returner
    

    def namedEntities (self, sentence):
    # Calculates the Named Entity feature for a single sentence. 
    # 
    #
    
        nlp_sentence = self.nlp(sentence)
        count = 0

        for ent in nlp_sentence.ents: # NORP (nationalities, religious and political groups)
            if ent.label_ == "PERSON":# FAC (buildings, airports etc.) 
                count += 1            # GPE (countries, cities etc.) TOEVOEGEN????????????
            if ent.label_ == "ORG":
                count += 1
                
        return count

    def pos_features (self, sentence):
    # Calculates various features based on the POS tags in a sentence
    #
    #
    
        nlp_sentence = self.nlp(sentence)
        pos_tags = []

        for token in nlp_sentence:
            pos_tags.append(token.pos_)

        counter = Counter(pos_tags)

        nouns = counter["NOUN"]
        adjectives = counter["ADJ"]
        numbers = counter["NUM"]
        det = counter["DET"]
        verbs = counter["VERB"]

        return nouns, adjectives, numbers, det, verbs

      
                
def main ():
# Main function that creates all the features for the database.
# A new csv file is created containing all the features. 
#

    start_time = time.time()
    run = features()

    sent = run.sentiment_feature 
    ner = run.ner_feature
    length = run.length_feature
    exclamation = run.exclmark_feature
    question = run.quesmark_feature
    contradictions = run.contra_feature
    temporal = run.temp_feature
    nouns = run.nouns_feature
    adjectives = run.adjectives_feature
    numbers = run.number_feature
    dets = run.det_feature
    verbs = run.verb_feature
    #domestic = run.domestic
    #foreign = run.foreign
    #politics = run.politics
    

    label = run.label
    frame = pd.DataFrame(sent, columns=["sentiment_score"])
    frame["ner"] = ner
    frame["length"] = length
    frame["excl_marks"] = exclamation
    frame["ques_marks"] = question
    frame["contradictions"] = contradictions
    frame["temporal"] = temporal
    frame["nouns"] = nouns
    frame["adjectives"] = adjectives
    frame["verbs"] = verbs
    frame["determiners"] = dets
    frame["numbers"] = numbers
    #frame["domestic"] = domestic
    #frame["foreign"] = foreign
    #frame["politics"] = politics


    frame["label"] = label
    frame.to_csv('features.csv', index=False)

    end_time = time.time()
    print("Execution finished in ",round(end_time - start_time,1), "seconds")

    
    

main()




            
