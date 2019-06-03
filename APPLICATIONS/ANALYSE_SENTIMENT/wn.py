# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:37:47 2019

@author: Juliette
"""
'''
from nltk.corpus import wordnet

syn = wordnet.synsets("awesome")
#synset
print(syn[0].name)

#trim down to word
print(syn[0].lemmas()[0].name())
'''
synonyms =[]
antonyms =[]

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
#print (set(synonyms))
#print (set(antonyms))

#similarity
w1= wordnet.synset("success.n.01")
w2= wordnet.synset("achievement.n.01")

print(w1.wup_similarity(w2))  

          
