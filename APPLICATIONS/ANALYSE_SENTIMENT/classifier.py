# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:10:59 2019
for future ref: poss to make into a class & use pickle for the rest
@author: Juliette
"""

import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import  LinearSVC, NuSVC

from sklearn.neighbors import KNeighborsClassifier

from nltk.classify import ClassifierI
from statistics  import mode 

class VoteClassifier (ClassifierI): #inherits from CI
    
    def __init__(self, *classifiers):
        self._classifiers=classifiers
    def classify (self,features):
        votes = []
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
            return mode(votes) #returns which algo got the most votes
    
    def confidence (self,features):
         votes = []
         for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
            cat_votes=votes.count(mode(votes)) #nb of votes / chosen category
            confidence = cat_votes/len(votes) 
            return confidence

short_pos= open("C:/Users/Juliette/Desktop/SENTIMENT ANALYSIS/positive.txt","r").read()
short_neg= open("C:/Users/Juliette/Desktop/SENTIMENT ANALYSIS/negative.txt","r").read()



documents= []
#pos = part of speech
#ici on determine trois grandes categories de mot et on ne retient que les mots y appartenant

allowed_word_type =["J","R","V"]
all_words= []
for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = nltk.word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos :
        if w[1][0] in allowed_word_type:
            all_words.append(w[0].lower())
            
for p in short_neg.split('\n'):
    documents.append((p,"neg"))
    words = nltk.word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos :
        if w[1][0] in allowed_word_type:
            all_words.append(w[0].lower())

#save_documents = open("documents.pickle","wb")
#pickle.dump(documents, save_documents)
#save_documents.close()

all_words=nltk.FreqDist(all_words) #here if we use 'most-common', we'll get stopwords : treatment here ?


print(all_words["hate"])

word_features =list(all_words.keys())[:5000] #top 3000 most prominent words
#save_word_features = open("word_features5k.pickle","wb")
#pickle.dump(word_features, save_word_features)
#save_word_features.close()

def find_features(document):
    words = nltk.word_tokenize(document) #getting all the existing words  in a set
    features= {}
    for w in word_features:
        features[w]=(w in words)
        
    return features

#featuresets_f = open("featuresets.pickle", "rb")
#featuresets = pickle.load(featuresets_f)
#featuresets_f.close()

#print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets= [(find_features(rev),category)for (rev,category)in documents]

save_file = open("featuresets.pickle","wb")
pickle.dump(featuresets,save_file)
save_file.close()

random.shuffle(featuresets)
print(len(featuresets))

#positive
training_set =featuresets[:10000]
test_set = featuresets[10000:]

#negative
#training_set=featuresets[:100]
#test_set=featuresets[100:]
##Naive Bayes algo : posterior= prior occurences * likelihood/evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)
print(" (nltk/original) NBA accuracy :",nltk.classify.accuracy(classifier,test_set)*100)
classifier.show_most_informative_features(10)    

#Multinomial NB
MNB_classifier =SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print(" MNBA accuracy :",nltk.classify.accuracy(MNB_classifier,test_set)*100)
#
#
#
#BNB_classifier =SklearnClassifier(BernoulliNB())
#BNB_classifier.train(training_set)
#print(" BNBA accuracy :",nltk.classify.accuracy(BNB_classifier,test_set)*100)
#
#KNN_classifier =SklearnClassifier(KNeighborsClassifier())
#KNN_classifier.train(training_set)
#print(" KNN accuracy :",nltk.classify.accuracy(KNN_classifier,test_set)*100)
# 
#
#LogReg_classifier =SklearnClassifier(LogisticRegression())
#LogReg_classifier.train(training_set)
#print(" Logistic Regression accuracy :",nltk.classify.accuracy(LogReg_classifier,test_set)*100)
#
#SGD_classifier =SklearnClassifier(SGDClassifier())
#SGD_classifier.train(training_set)
#print(" SGD accuracy :",nltk.classify.accuracy(SGD_classifier,test_set)*100)
#
#
# 
#
#LinSVC_classifier =SklearnClassifier(LinearSVC())
#LinSVC_classifier.train(training_set)
#print("Linear SVC accuracy :",nltk.classify.accuracy(LinSVC_classifier,test_set)*100)
#
#
#NuSVC_classifier =SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#print(" NuSVC accuracy :",nltk.classify.accuracy(NuSVC_classifier,test_set)*100)
#
#voted_classifier = VoteClassifier (classifier,MNB_classifier,BNB_classifier,LogReg_classifier,
#                                   SGD_classifier,LinSVC_classifier,NuSVC_classifier)
#print(" Voted classifier accuracy :",nltk.classify.accuracy(voted_classifier,test_set)*100)
#
#print("Classification =",voted_classifier.classify(test_set[0][0]), "Confidence :",voted_classifier.confidence(test_set[0][0])*100)

#def sentiment(text):
#    feats= find_features(text)
#    return voted_classifier.classify(feats)

##♣saving training set for classifier

#save_classifier =open("naivebayes.pickle","wb") #write in bytes nec.
#pickle.dump (classifier, save_classifier)
#save_classifier.close()

#save_classifier = open("BNBA.pickle","wb")
#pickle.dump(BNB_classifier,save_classifier)
#save_classifier.close()
#
#save_classifier1 = open("KNN.pickle","wb")
#pickle.dump(KNN_classifier,save_classifier1)
#save_classifier1.close()
#
#save_classifier2 = open("LogReg.pickle","wb")
#pickle.dump(LogReg_classifier,save_classifier2)
#save_classifier2.close()
#
#save_classifier3 = open("SGD.pickle","wb")
#pickle.dump(SGD_classifier,save_classifier3)
#save_classifier3.close()
#
#save_classifier4 = open("LinSVC.pickle","wb")
#pickle.dump(LinSVC_classifier,save_classifier4)
#save_classifier4.close()
#
#save_classifier5 = open("NuSVC.pickle","wb")
#pickle.dump(NuSVC_classifier,save_classifier5)
#save_classifier5.close()