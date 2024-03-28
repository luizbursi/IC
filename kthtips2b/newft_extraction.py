import numpy as np 
import os, sys
import cv2
import matlab.engine
import pickle 

eng = matlab.engine.start_matlab()

path_kthtips2b = "C:/Users/oluiz/OneDrive/Documentos/IC/Bases/kthtips2b/"

dirs_kthtips2b = [file for file in os.listdir(path_kthtips2b) if file.endswith('.png')]

feas_matrix = []
class_matrix = []
    

#vetor de classes
for file in dirs_kthtips2b:
    classe = file.split("_")
    classe = classe[0]
    class_matrix.append(classe)
    
path_pkl = "C:/Users/oluiz/OneDrive/Documentos/IC/Pickles/kthtips2b/"
typefile_f = "_kthtips2bFEAS.pkl"
typefile_c = "_kthtips2bCLASS.pkl"

  
