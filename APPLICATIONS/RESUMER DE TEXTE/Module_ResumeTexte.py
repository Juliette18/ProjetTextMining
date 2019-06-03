# -*- coding: utf-8 -*-
"""
@author: et8ge
"""


# Téléchargement des modules de NLTK

import nltk # utile au preprocessing des données
#nltk.download('punkt') # <=>sentence tokenizer
#nltk.download('stopwords') # Pour les stopwords
import bs4 as bs # Permet d'enlever les balises html et xml d'un texte
import urllib.request # permet de scrapper les données sur le net
import re # Permet d'utiliser les expression régulière
import heapq # Permet d'extraire les plus hautes valeurs d'une liste

"""
Fonction scraping()
@auteur :
                Skald
@parametre :
                string(link) -> lien URL de préférence wikipedia
@process :
                scrape tout le texte contenu sur une page web, 
                nettoie le texte des différentes balises html et xml (sauf les <p>),
                agrège uniquement le texte inscrit dans les balises paragraphes
@retour :
                string(article) -> texte dépouillé de tout markup
"""
def scraping(link) :
    scrapedData = urllib.request.urlopen(link)
    text = scrapedData.read()
    bsText = bs.BeautifulSoup(text,'lxml')
    paragraphes = bsText.find_all('p')

    article = ""
    for p in paragraphes:
        article += p.text
    
    return(article)

"""
Fonction preprocessing()
@auteur :
                Skald
@parametre :
                string(article) -> texte à résumer
                string(langage) -> langage du texte article
@process :
                retire toutes les référence propres à la documentation de wikipedia,
                si le texte est en anglais, retire les caractères spéciaux
@retour :
                string(article) -> texte nettoyer de tout caractère parasite
"""
def preprocessing(article, langage) :
    article = re.sub(r'\[[0-9]*\]', ' ', article)
    article = re.sub(r'\s+', ' ', article)
    
    if(langage == 'english') :
        articleAnglais = re.sub('[^a-zA-Z]', ' ', article)
        articleAnglais = re.sub(r'\s+', ' ', articleAnglais)
        return(articleAnglais)
        
    return(article)

"""
Fonction frequenceMots()
@auteur :
                Skald
@parametre :
                string(article) -> texte à résumer
@process :
                tokenisation du texte,
                incrémentation en fonction de la fréquence d'apparition des mots dans le texte,
                calcul du score des mot par l'opération fréquence du mot / fréquence max parmi tout les mots
@retour :
                List(freqMots) -> dictionnaire des mots conservés dans le texte, associés à leur fréquence d'apparition
"""
def frequenceMots(article) :
    freqMots = {}
    for word in nltk.word_tokenize(article):
        if word not in stopwords:
            if word not in freqMots.keys():
                freqMots[word] = 1
            else:
                freqMots[word] += 1

    maxFreq = max(freqMots.values())

    for word in freqMots.keys():
        freqMots[word] = (freqMots[word]/maxFreq)
    
    return(freqMots)

"""
Fonction frequenceMots()
@auteur :
                Skald
@parametre :
                List(freqMots) -> dictionnaire des mots du texte, associés à leur fréquence d'apparition
                int(tailleMaxPhrase) -> taille maximale des phrases qui seront ensuite sélectionnées pour apparaitre au sein du résumé
@process :
                additionne les scores des mots contenu dans une phrase,
                et ce afin de former le socre de cette dernière
@retour :
                List(scorePhrase) -> dictionnaire des phrases composant le texte, associés à leur score
"""
def scoreParPhrase(freqMots, tailleMaxPhrase) :
    if(tailleMaxPhrase > 30) :
        tailleMaxPhrase = 20
    
    scorePhrase = {}
    for phrase in phraseList:
        for word in nltk.word_tokenize(phrase.lower()):
            if word in freqMots.keys():
                if len(phrase.split(' ')) < tailleMaxPhrase:
                    if phrase not in scorePhrase.keys():
                        scorePhrase[phrase] = freqMots[word]
                    else:
                        scorePhrase[phrase] += freqMots[word]
    
    return(scorePhrase)

"""
Fonction BuildResume()
@auteur :
                Skald
@parametre :
                Liste(scorePhrase) -> dictionnaire des phrases composant le texte, associés à leur score
                int(longResume) -> taille du résumé en nombre de phrases qui le compose
@process :
                Sélectionne les X phrases disposant du meilleure score,
                puis forme un résumé à partir de celles-ci
@retour :
                string(resume) -> résumé de l'article original, composé des meilleures phrases
"""
def BuildResume(scorePhrase, longResume) :
    BestPhrases = heapq.nlargest(longResume, scorePhrase, key=scorePhrase.get)
    resume = ' '.join(BestPhrases)
    return(resume)

#########################################MAIN##############################################
#placer le lien du document ici
link='https://fr.wikipedia.org/wiki/Informatique'
#Entrer la langue du texte (en anglais, french pour francais par exemple)
langage = 'french'
#Entrer la taille maximale (en nombre de mots) des phrases qui apparaitront dans le résumé
tailleMaxPhrase = 20
#Entrer la longueur maximale (en nombre de phrase) du résumé de texte final
longResume = 5

#running
article = scraping(link)

article = preprocessing(article, langage)

phraseList = nltk.sent_tokenize(article)

stopwords = nltk.corpus.stopwords.words(langage)

freqMots = frequenceMots(article)

scorePhrase = scoreParPhrase(freqMots, tailleMaxPhrase)

resume = BuildResume(scorePhrase, longResume)

print(resume)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        