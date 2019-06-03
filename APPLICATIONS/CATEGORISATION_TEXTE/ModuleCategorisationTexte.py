# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:42:09 2019

@author: et8ge
"""

""" IMPORTATION """
#Importation des bibliothèque utilitaire
from sklearn.model_selection import KFold
import pickle
import os
import re

#Importation du dataset
from sklearn.datasets import fetch_20newsgroups

#Importation des outils de prétraitment de texte
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Importation des différents modèle, pouvant constituer le métaclassifieur
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

#Importation du métaclassifieur
from sklearn.ensemble import VotingClassifier

""" VARIABLES GLOBALES """

categories_initiales = ['comp.graphics',
                        'comp.os.ms-windows.misc',
                        'comp.sys.ibm.pc.hardware',
                        'comp.sys.mac.hardware',
                        'comp.windows.x',
                        'rec.autos',
                        'rec.motorcycles',
                        'rec.sport.baseball',
                        'rec.sport.hockey',
                        'sci.crypt',
                        'sci.electronics',
                        'sci.med',
                        'sci.space',
                        'misc.forsale',
                        'talk.politics.misc',
                        'talk.politics.guns',
                        'talk.politics.mideast',
                        'talk.religion.misc',
                        'alt.atheism',
                        'soc.religion.christian']

categories = []

serial_folder = "Serial_Folder"

serial_format = ".sf"


""" FACTORISATION """

def metaclassifieur() :
    metaClf = VotingClassifier(estimators=[('CLf1',MultinomialNB(
                                                                    alpha          = 0.035)),

                                           ('Clf2',SGDClassifier(
                                                                    loss           = 'hinge',
                                                                    penalty        = 'l2',
                                                                    alpha          = 1e-3,
                                                                    random_state   = 42,
                                                                    max_iter       = 100,
                                                                    tol            = None)),

                                           ('Clf3',KNeighborsClassifier(
                                                                    n_neighbors    = 1,
                                                                    weights        = 'uniform')),

                                           ('Clf4',DecisionTreeClassifier()),

                                           ('Clf5',RandomForestClassifier()),

                                           ('Clf6',LogisticRegression()),

                                           ('Clf7',SVC(
                                                                    kernel         = 'linear'))],

                                voting="hard")
    return(metaClf)


def load_And_Split_20NewsGroup(categories) :
    dataset = fetch_20newsgroups(subset="all", categories = categories, remove =('headers', 'footers', 'quotes'), shuffle = True)

    X_Train = dataset.data
    Y_Train = dataset.target
    labels = dataset.target_names

    return(X_Train, Y_Train, labels)


def fitting_By_CV(X_Train_Processed, Y_Train, metaClf) :

    i=0
    print("Apprentissage en cours, veuillez patienter... "+str(i)+"%")
    print("")

    cv = KFold(n_splits = 5)

    score = 0

    for train, test in cv.split(X_Train_Processed, Y_Train):
        metaClf.fit(X_Train_Processed[train], Y_Train[train])
        score += metaClf.score(X_Train_Processed[test], Y_Train[test])
        i+=20
        print("Apprentissage en cours, veuillez patienter... "+str(i)+"%")
        print("")

    score /= 5
    return (score, metaClf)

def serialisation(nom, objet) :
    pickle_out = open(serial_folder + "/" + nom + serial_format, "wb")
    pickle.dump(objet, pickle_out)
    pickle_out.close()

def deserialisation(nom) :

    pickle_in = open(serial_folder + "/" + nom + serial_format, "rb")
    objet = pickle.load(pickle_in)
    pickle_in.close()
    return(objet)

def serialisation_du_modele(metaClf, count_vect, tfidf_transformer) :

    serialisation('classifier', metaClf),
    serialisation('vectorizer', count_vect),
    serialisation('tfIdfer', tfidf_transformer)
    serialisation('categories', categories)

def deserialisation_du_modele() :

    metaClf = deserialisation('classifier')
    count_vect = deserialisation('vectorizer')
    tfidf_transformer = deserialisation('tfIdfer')
    return (metaClf, count_vect, tfidf_transformer)

def le_modele_n_existe_pas() :
    Creation_eventuelle_du_serial_folder()

    if (os.path.exists(serial_folder + "/" + "classifier" + serial_format) and
       os.path.exists(serial_folder + "/" + "vectorizer" + serial_format) and
       os.path.exists(serial_folder + "/" + "tfIdfer" + serial_format) and
       os.path.exists(serial_folder + "/" + "categories" + serial_format)) :
           last_categories = deserialisation('categories')
           if (last_categories.sort() == categories.sort()) :
               return(False)
           else :
               return(True)
    else :
        return(True)

def Creation_eventuelle_du_serial_folder() :
    if not (os.path.exists(serial_folder)) :
        os.makedirs(serial_folder)


""" PATHS """
def execution():

    #Désérialisation du modele
    metaClf, count_vect, tfidf_transformer = deserialisation_du_modele()

    article = input_article()

    if (article == "demo"):
        X_New = ['God bless America.',
                 'I compute faster than the GPU of my PC.',
                 'Guilhem is a cancer.',
                 'My MAC is so beautiful !',
                 'Bob does not believe in God.'] #(en demo) Pour retomber sur le christianisme, il suffit de remplacer "Bob" par "I"...
        demo_analyse(X_New, metaClf, count_vect, tfidf_transformer)

    else :
        X_New = [article]
        analyse(X_New, metaClf, count_vect, tfidf_transformer)

def input_article() :
    print("Veuillez à présent entrer du texte afin que l'algorithme puisse prédire son thème.")
    print("Le texte se devra d'être écrit en anglais et correspondre aux labels initialement paramétrés,")
    print("Auquel cas le résultat sera biaisé.")
    print("Entrer 'demo' pour accéder à des phrases types.")

    article = input("Votre article >>")
    print("")

    return(article)

def analyse(X_New, metaClf, count_vect, tfidf_transformer) :
    X_New_Counts = count_vect.transform(X_New)

    X_New_Processed = tfidf_transformer.transform(X_New_Counts)

    predicted = metaClf.predict(X_New_Processed)

    for doc, category in zip(X_New, predicted):
        print(doc)
        print("-> Theme de l'article : " + labels[category])
    print("")

def demo_analyse(X_New, metaClf, count_vect, tfidf_transformer) :
    X_New_Counts = count_vect.transform(X_New)

    X_New_Processed = tfidf_transformer.transform(X_New_Counts)

    predicted = metaClf.predict(X_New_Processed)

    for doc, category in zip(X_New, predicted):
        print('%r -> %s' % (doc, labels[category]))
    print("")

def creation_du_modele():
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_Train) #return sparse matrix, [n_samples, n_features]

    tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_Train_Processed = tfidf_transformer.transform(X_train_counts) #return sparse matrix, [n_samples, n_features]

    metaClf = metaclassifieur()

    ModelScore, metaClf = fitting_By_CV(X_Train_Processed, Y_Train, metaClf)
    print("Taux de réussite du model : " + str(round(ModelScore*100,2)) + "%.")
    print("")

    serialisation_du_modele(metaClf, count_vect, tfidf_transformer)

def presentation_du_programme() :
    print("Bonjour,")
    print("Bienvenue sur le module de catégorisation de texte.")
    print("")
    print("Ce programme, développé par Gérome FERRAND, s'inscrit dans un projet plus large qui réunit d'autres modules de TEXT MINING.")
    print("Ces travaux ont été conçu dans le cadre du projet tutoré de l'ASPE DUT INFO 2018/2019 de l'IUT la Doua à l'Université LYON1.")
    print("Ils ont été mis au point par :")
    print("Alexandre DUFOUR")
    print("Gérome FERRAND")
    print("Jeremy LAURENT")
    print("Nathan MOUSSU")
    print("Juliette VATTON")
    print("")


def texte_parametrage_du_modele():
    print("Avant de commencer à utiliser ce programme,")
    print("Merci de bien vouloir le paramétrer en fonction de vos besoins.")
    print("")
    print("Pour cela, veuillez taper la valeur associée au.x thèmes/groupe de thèmes que vous souhaitez soumettre au modèle.")
    print("Finissez par une entrée vide (i.e. appuyer sur la touche entrer, sans valeur) pour terminer la sélection personalisée.")
    print("")
    print("1 : comp.graphics")
    print("2 : comp.os.ms-windows.misc")
    print("3 : comp.sys.ibm.pc.hardware")
    print("4 : comp.sys.mac.hardware")
    print("5 : comp.windows.x")
    print("6 : rec.autos")
    print("7 : rec.motorcycles")
    print("8 : rec.sport.baseball")
    print("9 : rec.sport.hockey")
    print("10 : sci.crypt")
    print("11 : sci.electronics")
    print("12 : sci.med")
    print("13 : sci.space")
    print("14 : misc.forsale")
    print("15 : talk.politics.misc")
    print("16 : talk.politics.guns")
    print("17 : talk.politics.mideast")
    print("18 : talk.religion.misc")
    print("19 : alt.atheism")
    print("20 : soc.religion.christian")

    print("")

    print("all : Pour ceux qui ont du temps à perdre...,")
    print("Ce set contient tout les thèmes.")

    print("")

    print("demo : Pour les démonstrations,")
    print("Ce set contient les thèmes suivants :")
    print("-sci.med")
    print("-comp.sys.ibm.pc.hardware")
    print("-comp.sys.mac.hardware")
    print("-alt.atheism")
    print("-soc.religion.christian")

    print("")

    print("standard : Pour une utilisation standard, (sélection par défaut en mode script)")
    print("Ce set contient les thèmes suivants :")
    print("-comp.graphics")
    print("-comp.sys.ibm.pc.hardware")
    print("-comp.sys.mac.hardware")
    print("-rec.autos")
    print("-rec.motorcycles")
    print("-rec.sport.baseball")
    print("-rec.sport.hockey")
    print("-sci.crypt")
    print("-sci.electronics")
    print("-sci.med")
    print("-sci.space")
    print("-talk.politics.misc")
    print("-talk.religion.misc")

    print("")

    print("Vous pouvez également entrer 'show' afin de consulter les thèmes choisis.")
    print("ou alors 'del' pour entrer en mode suppression.")

    print("")

def selection_des_categories() :
    global categories

    while True :
        choix = str(input("input >> "))
        print("")

        if (choix == "show") :
            if (len(categories) == 0) :
                print("Vous n'avez sélectionné aucun thème.")
            else :
                print("Pour l'instant, vous avez sélectionné les thèmes suivants :")
                i=0
                for theme in categories :
                    i=i+1
                    print(str(i) + "-" + theme)
                print("")

        elif (choix == "del") :
            while True :
                if (len(categories) == 0) :
                    print("Il n'y a aucun thème à supprimer.")
                    break
                else :
                    print("Pour l'instant, vous avez sélectionné les thèmes suivants :")
                    i=0
                    for theme in categories :
                        i=i+1
                        print(str(i) + "-" + theme)
                    print(len(categories))
                    print("")

                    print("Quel thème voulez-vous supprimer ? (taper 'ok' pour sortir du mode suppression.)")

                    suppr = str(input("input >> "))
                    print("")

                    if (re.match("^[0-9]{1,2}$", suppr)) :
                        if (int(suppr) <= len(categories) and int(suppr) > 0) :
                            print("Vous avez supprimer le thème suivants : " + categories.pop(int(suppr)-1))
                            print("")
                        else :
                            print("ce nombre ne correspond pas au valeurs indiquées.")
                            print("")
                    elif (suppr == "ok") or (len(categories)==0) :
                        print("Vous pouvez continuer votre sélection ou taper sur entrer pour passer à la suite.")
                        break
                    else :
                        print("L'entrée n'est pas valide, veuillez la réitérer.")
                        print("")


        elif choix == "all" :
            categories = [  'comp.graphics',
                            'comp.os.ms-windows.misc',
                            'comp.sys.ibm.pc.hardware',
                            'comp.sys.mac.hardware',
                            'comp.windows.x',
                            'rec.autos',
                            'rec.motorcycles',
                            'rec.sport.baseball',
                            'rec.sport.hockey',
                            'sci.crypt',
                            'sci.electronics',
                            'sci.med',
                            'sci.space',
                            'misc.forsale',
                            'talk.politics.misc',
                            'talk.politics.guns',
                            'talk.politics.mideast',
                            'talk.religion.misc',
                            'alt.atheism',
                            'soc.religion.christian']
            break

        elif choix == "demo" :
            categories = [  'comp.sys.ibm.pc.hardware',
                            'comp.sys.mac.hardware',
                            'sci.med',
                            'alt.atheism',
                            'soc.religion.christian']
            break

        elif choix == "standard" :
            categories = [  'comp.graphics',
                            'comp.sys.ibm.pc.hardware',
                            'comp.sys.mac.hardware',
                            'rec.autos',
                            'rec.motorcycles',
                            'rec.sport.baseball',
                            'rec.sport.hockey',
                            'sci.crypt',
                            'sci.electronics',
                            'sci.med',
                            'sci.space',
                            'talk.politics.misc',
                            'talk.religion.misc']
            break

        elif re.match("^[1-9]$|^1[0-9]$|^20$", choix) :
            print("Vous avez sélectionné :")
            print(categories_initiales[int(choix)-1])
            if ((categories.count(categories_initiales[int(choix)-1]))==0) :
                categories.append(categories_initiales[int(choix)-1])
            else :
                print("vous avez deja ajouté ce thème !")
            continue

        elif choix == "" :
            if (len(categories)) == 0 :
                print("Vous n'avez sélectionné aucune categorie, veuillez réitérer.")
                continue

            else :
                break

        else :
            print("L'entrée n'est pas valide, veuillez la réitérer.")

    print("Vous avez sélectionné les thèmes suivants :")
    for themes in categories :
        print("-" + themes)
    print("")
    return(categories)

def parametrage_du_modele() :

    texte_parametrage_du_modele()

    categories = selection_des_categories()

    return(categories)


""" MAIN """

presentation_du_programme()

categories = parametrage_du_modele()

#categories = ['sci.med', 'alt.atheism', 'soc.religion.christian', 'comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware']

X_Train, Y_Train, labels = load_And_Split_20NewsGroup(categories)

if (le_modele_n_existe_pas()) :
    creation_du_modele()

execution()
























