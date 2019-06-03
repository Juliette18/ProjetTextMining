# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:02:03 2019

@author: LAURENT Jérémy
"""
import sys
import nltk
import sklearn
import pandas
import numpy

#Importation de Scikit-learn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ====================================DEBUT FONCTION=================================== #
def RegularExpression(text_messages):
    # Procédure qui remplace des adresses mails par 'emailaddress'
    processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
    # Procédure qui remplace les URLs par 'webaddress'
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
    # Procédure qui remplace les sympboles d'argents par 'moneysymb
    processed = processed.str.replace(r'£|\$', 'moneysymb')
    # Procédure qui remplace les numéros de téléhpone par 'phonenumbr' 
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
    # Procédure qui remplace les chiffres/nombres par 'numbr'
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
    # Suppression de la ponctuation
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    # Remplacement des espaces multiples par des espaces simples
    processed = processed.str.replace(r'\s+', ' ')
    # Mise en minuscule des mots
    processed = processed.str.lower()
    return processed

def StopWord(processed):
    # Suppression des mots-d'arrêts
    stop_words = set(stopwords.words('english'))
    processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    return processed

def Stemming(processed):
    ps = nltk.PorterStemmer()
    processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
    return processed
    
def find_features(message):
    words = word_tokenize(str(message))
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# ====================================FIN FONCTION=================================== #
    

# ====================================DEBUT MAIN=================================== #
# Affichage des versions des bibliothèques
print('Python: {}'.format(sys.version))
print('NLTK: {}'.format(nltk.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy: {}'.format(numpy.__version__))
print('')
print("Bienvenue dans le programme de détection de SPAM Version 1")
print("Veuillez patienter...")
print('')
# ============== DEBUT APPRENTISSAGE============== #
# Chargement des messages types SMS
df = pd.read_table('SMSSPamCollection', header=None, encoding='utf-8')
# Conserve la première cologne de la matrice
classes = df[0]
# Convertir les labels de la classes en valeur binaire:
# 0 = ham and 1 = spam et les mettres dans une liste Y
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
# Stockage des SMS dans la variable text_messages
text_messages = df[1]
processed = RegularExpression(text_messages)
processed = StopWord(processed)
processed = Stemming(processed)
# Création du Bag-of-words
bag = []

# On met les mots dans le Bag
for message in processed:
    words = word_tokenize(message)
    for w in words:
        bag.append(w)

bag = nltk.FreqDist(bag)
# Utilisation des 1500 mots les plus communs comme références
word_features = list(bag.keys())[:1500]
# Automatision de la fonction find_feature
messages = list(zip(processed, Y))
# Echantillonnage des données
seed=1
np.random.seed = seed
np.random.shuffle(messages)
featuresets = [(find_features(text), label) for (text, label) in messages]
# Répartition: 75% pour le training set et 25 % pour le testing set
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
training_n2=training
# Modèle individuels optimisés
KNC=KNeighborsClassifier(n_neighbors=1)
DTC=DecisionTreeClassifier()
RFC=RandomForestClassifier(n_estimators=100)
LG=LogisticRegression(solver='saga')
SGDC=SGDClassifier(max_iter = 100)
M=MultinomialNB(alpha=0.4)
SC=SVC(kernel = 'linear')
# Construction du meta-modèle
meta_model2 = SklearnClassifier(VotingClassifier(estimators=[('k1',KNC),('k2',DTC),('k3',RFC),('k4',LG),('k5',SGDC),('k6',M),('k7',SC)],voting="hard"))
meta_model2.train(training_n2)
# ============== FIN APPRENTISSAGE  ============== #

while True:
    print("Merci de copier-coller le texte qui vous semble être un SPAM")
    TEXTE = input()
    TEXTE = pd.Series(TEXTE)
    TEXTE = RegularExpression(TEXTE)
    TEXTE = StopWord(TEXTE)
    TEXTE = Stemming(TEXTE)
    TEXTE = [TEXTE]
    messages = list(zip(TEXTE, Y))
    featuresets = [(find_features(text), label) for (text, label) in messages]
    txt_features, labels = zip(*featuresets)
    prediction = meta_model2.classify_many(txt_features)
    if (prediction == [0]):
        print("Votre message semble être un message normal.")
    else:
        print("C'est probablement un SPAM !!")
    print("\n")
    print("Voulez-vous quitter le programme ?")
    reponse=input()
    if (reponse == "oui" or reponse == "OUI"):
        break
        print("Au revoir")
    else:
        print('')
# ====================================FIN MAIN=================================== #
