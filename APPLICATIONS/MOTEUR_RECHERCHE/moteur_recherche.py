# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:08:30 2018

@author: Alex Dufour
"""


# Librairies
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups


# "Hyperparamètres"
stop_words = set(stopwords.words('english'))
stop_words.update(stop_words,{'.',',','!','?','\'s', '<', '>', ':', ';', '/', '(', ')', '-', '_', '{', '}', '--', '...'})
lmtzr=WordNetLemmatizer()
ps = PorterStemmer()
nbTop = 10;


# Travail préliminaire ########################################################


def split(text):
    """ Fonction qui prend en paramètre un texte (chaine de caractère) et qui 
    renvoie la liste de ses mots ayant été filtrés puis ayant subits certaines 
    transformations: lemming puis stemming """
    
    # Tokenisation ************************************************************
    # Découpage du texte en mots (words est une liste de chaine de caractère)
    words = word_tokenize(text)

    # Stop-words **************************************************************
    # Filtrage des mots : on supprime de words ceux qui sont contenus dans 
    # stop_words car ils sont supposés avoir trop peu de sens.
    # (mots trop communs, mots de liaisons, ponctuation)
    words_clean = []
    for word in words:
        if word.lower() not in stop_words:
            words_clean.append(word)
    
    # Lemming *****************************************************************
    # Transformation des mots en un unique mot-clé (lemme) les représentant. 
    # Ex: divided, dividing, divided, divides -> divide
    words_lemmed = [lmtzr.lemmatize(word) for word in words_clean]
            
    # Stemming ****************************************************************
    # Transformation des mots en un unique radical les représentant. 
    # Ex: divided, dividing, divided, divides -> divid
    words_stemmed = [ps.stem(word) for word in words_lemmed]
    
    return words_stemmed
# End


def count(words, wordbase):
    """ Fonction qui prend en paramètre un texte 'splité' en mots et la liste 
    des mots du corpus, et renvoie le vecteur contenant le nombre d'occurence 
    dans le texte des mots du corpus."""
    
    vector = np.zeros(len(wordbase))
    for i in range(len(wordbase)):
        if wordbase[i] in words:
            vector[i] += 1
    return vector
# End
    

def vectorisation(text, wordbase):
    """ Fonction qui prend en paramètre un texte sous la forme d'une chaine de 
    caractère, et la liste des mots du corpus et qui renvoie le vecteur 
    représentant le texte dans la base du corpus."""
    
    return count(split(text), wordbase)
# End


def preliminaryWork(corpus):
    """ Fonction qui prend en paramètre un corpus de texte sous la forme d'une 
    liste de chaines de caractère, et qui renvoie la liste des mots utilisés 
    comme base pour représenter les textes qui le compose, et la matrice de ces
    textes dans cette base, les coordonnées étant calculées par la formule du 
    TF-IDF."""
    
    # Découpage, tri et transformation des textes (voir split)
    corpus_words = []
    for i in range(len(corpus)):
        corpus_words.append(split(corpus[i]))
        
    # Construction de la liste des mots du corpus (intersection des mots des 
    # textes). Wordset est un objet de type set, intéressant car il permet de 
    # faire l'intersection seul, mais pas ordonné. On construit donc wordbase
    # à partir des mots de wordset pour pouvoir associer 1 mot à 1 coordonnée.
    wordset = set()
    for words in corpus_words:
        wordset = wordset.union(set(words))
    wordbase = [word for word in wordset]
    
    # Construction de la matrice représentant les textes dans la base wordbase.
    # On l'initialise avec les vecteurs dont les coordonnées sont les 
    # occurences brutes.
    matrix = []
    for words in corpus_words:
        matrix.append(count(words, wordbase))
        
    # Calcul du nombe de textes contenant chaque mot
    nt = np.zeros(len(wordbase))
    for m in range(len(wordbase)):
        for line in matrix:
            if line[m] > 0:
                nt[m] += 1
    
    # Calcul du tf-idf de chaque mot dans chaque texte
    # TF(mot dans un texte) = log(1 + nb d'occurence de ce mot dans ce texte)
    # IDF(mot) = log(nombre de textes total / nombre de texte cntenant ce mot)
    # TF-IDF(mot dans un texte) = TF(mot dans un texte) * IDF(mot)
    for t in range(len(matrix)):
        for m in range(len(wordbase)):
            matrix[t][m] = math.log(1 + matrix[t][m]) * math.log(len(corpus) / nt[m])
            
    return (matrix, wordbase)
#End


###############################################################################

# Recherche ###################################################################

def scal(v1, v2):
    """ Fonction qui calcule le produit scalaire entre deux vecteurs de même
    taille et la renvoie."""
    
    scal = 0
    for i in range(len(v1)):
        scal += v1[i] * v2[i]
    return (scal)
#End
  
    
def norm(v):
    """ Fonction qui calcule la norme 2 d'un vecteur et la renvoie."""
    n = math.sqrt(scal(v, v))
    if (n == 0):
        n = 1
    return (n)
#End


def iMax(similarity):
    """ Fonction qui renvoie l'indice du maximum de la liste passée en 
    paramètre """
    imax = 0
    for i in range(1,len(similarity)):
        if (similarity[i] > similarity[imax]):
            imax = i
    return (imax)
#End


def top(similarity, nbTop):
    """ Fonction qui renvoie la liste des indices des nbTop éléments 
    les plus grands de la liste passée en paramètre (similarity), 
    dans l'ordre décroissant. """
    order = []
    cptTop = 0
    imax = iMax(similarity)
    while (similarity[imax] >= 0 and cptTop < nbTop):
        order.append(imax)
        # On met à -1 l'élément dont on vient de prendre l'indice pour ne plus le prendre en compte
        similarity[imax] = -1
        cptTop += 1
        imax = iMax(similarity)
    return (order)
#End


def research(request, matrix, wordbase):
    """ Fonction qui prend en paramètre une requête sous la forme d'une
    chaîne de caractère, la matrice représentatrice du corpus et la liste
    des tokens, et qui renvoie la liste des indices des textes dans le
    corpus correspondant le plus à la requête. """
    # Vectorisation de la requête
    vector = vectorisation(request, wordbase)
    # Calcul de la liste des score de similarité pour chaque vecteur de la matrice
    similarity = []
    for v2 in matrix:
        similarity.append(scal(vector, v2) / (norm(vector) * norm(v2)))
    # On renvoie les meilleurs résultats
    return (top(similarity, nbTop))
#End
    

###############################################################################

# Affichage (rudimentaire) des résultats ######################################
    

def printResearch(order, corpus):
    """ Fonction qui à partir des résultats d'une requête (liste d'indices) et 
    du corpus utilisé affiche les textes dans la console, le plus représentatif
    en premier"""
    print("*\t*\t*\t*\t*\t*\t*\t*")
    print("*\t*\tRésultats de la recherche\t*\t*")
    print("*\t*\t*\t*\t*\t*\t*\t*\n\n")
    input("Appuyer sur une touche...\n")
    for i in range(len(order)):
        print("*\t*\t*\t*\t*\t*\t*\t*")
        print("*\t*\tNuméro " + str(i + 1) + " - Texte " + str(order[i]) + "\t*\t*\t*")
        print("*\t*\t*\t*\t*\t*\t*\t*\n\n")
        print(corpus.data[order[i]])
        # Pause entre l'affichage de chaque texte
        input("Appuyer sur une touche...\n")
    print("Fin")
#End

# Exemple #####################################################################
    

# Corpus utilisé
corpus = fetch_20newsgroups()

# Nombre de texte qu'on prend en compte (optionnel)
nbTxt = 100

###############################################################################

# main

# Construction de la matrice
(matrix, wordbase) = preliminaryWork(corpus.data[:nbTxt])

request = input("Recherche : ")
while request != "exit":
    # Traitemtn d'une recherche
    order = research(request, matrix, wordbase)
    printResearch(order, corpus)
    request = input("Recherche : ")

#End