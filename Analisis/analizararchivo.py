import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

class AnalizarArchivo():
    def __init__(self) :
        pass

    def LeerArchivo(self,nombreArchivo):
        dataFrame = pd.read_csv(nombreArchivo);
        return dataFrame