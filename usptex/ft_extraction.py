# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:43:32 2023

@author: oluiz
"""


#código para a extração de dados de cada imagem da base de dados USPTEX
#irei atras da base de dados, analisar as features de cada imagem com matlab e aí irei armazena-las em um pickle
#em um outro código, irei pegar as features extraídas e classifica-las como venho fazendo 


#importando as bibliotecas
import numpy as np 
import os, sys
import cv2
import matlab.engine
import pickle 

#criação da classe para uso da engine de matlab em python 
eng = matlab.engine.start_matlab()

#path para a base de dados 
path_usptex = "C:/Users/oluiz/OneDrive/Documentos/IC/Bases/usptex/"

#cria um vetor com os arquivos da base de dados 
dirs_usptex = [file for file in os.listdir(path_usptex) if file.endswith('.png')]

#vetores que servirão para armazernar as features(de cada metodo descritivo) e as classes de cada imagem
feas_matrix = []
class_matrix = []
    

#vetor de classes
for file in dirs_usptex:
    classe = file.split("_")
    classe = classe[0]
    classe = classe.split("c")
    classe = classe[1]
    class_matrix.append(classe)
    
    
    
#path para o diretorio com os descritores de imagem
path_descriptors = "C:/Users/oluiz/OneDrive/Documentos/IC/Descritores/"
dirs_descriptors = os.listdir(path_descriptors) #lista dos metodos que serão utilizados


#variavel para facilitar a criação de um .pkl e para o diretorio dele 
path_pkl = "C:/Users/oluiz/OneDrive/Documentos/IC/Pickles/usptex/"
typefile_f = "_usptexFEAS.pkl"
typefile_c = "_usptexCLASS.pkl"
#as variaveis acima servem para podermos salvar de forma mais organizada os arquivos .pkl e facilitar o encontro do .pkl quando necessario 



#serão dois 'for' que irão englobar todas as imagens em todos os metodos descritores
for descriptor in dirs_descriptors:     #ele vai fazer o laço no diretório com todas as funções extratoras de features
    
    eng.addpath(path_descriptors + descriptor)  #adequação do caminho da função no módulo de matlab
    feas_matrix = []  #array que vai receber as características extraídas
    for images in dirs_usptex:  #laço que percorre as imagens dentro da base
        try:   
            img = cv2.imread(path_usptex+ images)  #parte de processamento de imagens
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #parte de processamento de imagens
            img = matlab.double(img.tolist()) #aplicação da função que adequa a imagem pra função
            feas = eng.run(img)    #aplicação da função que extrai as features da imagem
            feas_matrix.append(feas) #faz um append na matriz, para ser uma matriz pelas n imagens por m features
        except Exception as e:
            print(f"Erro ao processar a imagem {images} com o descritor {descriptor}: {e}")
            break  # Descomente esta linha se quiser sair do loop de imagens atual e tentar o próximo descritor
        
        
        
        
    #vai salvar no .pkl as features de todas as imagen
    data = {"descriptor": descriptor, "features": feas_matrix }
    with open(path_pkl+descriptor+typefile_f, "wb" ) as f:
        pickle.dump(data,f)
    #print(feas_matrix)
    
datac = {"class": class_matrix}
with open(path_pkl+typefile_c, "wb") as fc:
    pickle.dump(datac,fc)
eng.quit()

