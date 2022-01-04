import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import json as json

class AnalizarArchivo():
    def __init__(self) :
        pass

    def LeerArchivo(self,nombreArchivo):
        dataFrame = pd.read_csv(nombreArchivo);
        return dataFrame

    def LeerArchivoExcel(self,nombreArchivo):
        dataFrame = pd.read_excel(nombreArchivo);
        return dataFrame

    def LeerArchivoJson(self,nombreArchivo):
        dataFrame = pd.read_json(nombreArchivo,orient='index');
        #print(dataFrame.head())
        # load data using Python JSON module
       # print(type(dataFrame))
       
        return dataFrame    


        