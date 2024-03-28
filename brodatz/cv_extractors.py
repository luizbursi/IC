# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:13:06 2023

@author: oluiz
"""

#nesse arquivo irei abrir o pickle com as features extraidas no ft_extraction.py
#com os dados em mão, irei treinar e classificar as informações
#base de dados: 1200tex



import pickle 
import os, sys
import numpy as np 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import svm



#path para os .pkl
path_pkl = "C:/Users/oluiz/OneDrive/Documentos/IC/Pickles/brodatz/"
dirs_pkl = [file for file in os.listdir(path_pkl) if not file.endswith('CLASS.pkl')]
classepkl = [file for file in os.listdir(path_pkl) if file.endswith('CLASS.pkl')]
classepkl = classepkl[0]


#matriz com as classes 
with open(path_pkl+classepkl, "rb") as fc:
    classes = pickle.load(fc)
classes = classes["class"]
classes = np.array(classes)

#classificadores 
knn = KNeighborsClassifier(n_neighbors=1)
lda = LinearDiscriminantAnalysis()
svmc = svm.SVC()

#for ira rodar todos os pickles com as features
#fazer o treino e depois classificar 
for pkl in dirs_pkl:
    path = path_pkl+pkl
    with open(path, "rb") as f:
        data = pickle.load(f)
    descriptor = data["descriptor"]
    feas_matrix = data["features"]
    feas_matrix = np.array(feas_matrix)
    feas_matrix =feas_matrix.reshape(feas_matrix.shape[0], -1)
    cv_lda = cross_val_score(lda, feas_matrix,classes,cv=10)    
    cv_knn = cross_val_score(knn, feas_matrix,classes,cv=10)
    cv_svm = cross_val_score(svmc,feas_matrix,classes,cv=10 )
    print(descriptor,"->", np.mean(cv_lda),"/", np.mean(cv_knn),"/",np.mean(cv_svm))



