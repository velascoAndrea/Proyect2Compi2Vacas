from math import degrees
import numpy as np
from pandas.core.algorithms import mode
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
from datetime import date, datetime
from sklearn.metrics import mean_squared_error, r2_score

class Reportes():
    #Tendencia de la infección por Covid-19 en un País.
    def __init__(self) :
        pass

    def AnalizarRep1(self,nombreArchivo,pais,columnaPais,dependiente,independiente):

        dataFrame = pd.read_csv(nombreArchivo);
        is_country = dataFrame[columnaPais] == pais
        #print(is_country)
        dataFrameFiltrado = dataFrame[is_country]
        #pasar el formato a fecha
        fechas = pd.to_datetime(dataFrameFiltrado[independiente])
        
        #creo un nuevo DataFrame Para agruparlo por Fecha
        ds =pd.DataFrame()
        ds[independiente] = fechas
        ds[dependiente] = dataFrameFiltrado[dependiente] 
        ds = ds.groupby(by=independiente).sum()
        #print(ds.keys())
        #print(ds.index.values)
       
        #Fecha X Variable Independiente
        x = ds.index.values
        #Numero de Casos Y Dependiente
        y = ds[dependiente].values

        #Tengo que reordenar para que funcione
        x = x.reshape(len(x),1)
        print(x.shape,"DATO X")
        print(y.shape,"DATO Y")



        
        #--------------------------------------ANALISIS----------------------------------------------------------
        FuncPoli(x,y,pais)

        return ""

    
def  FuncPoli(x,y,pais):
    #Datos de Entrenamiento
    plt.scatter(x,y)
    x_train,x_test, y_train, y_test = train_test_split(x,y)
    grado = 2
    polynomial_reg = PolynomialFeatures(degree = grado)

    x_transform = polynomial_reg.fit_transform(x) # La mera X
    x_train_poli = polynomial_reg.fit_transform(x_train) # La X de entrenamiento
    x_test_poli = polynomial_reg.fit_transform(x_test) # La X test

    # fit the model principal
    model = linear_model.LinearRegression()
    model.fit(x_transform,y)
    y_new = model.predict(x_transform)

    #Ajuste con las pruebas de entrenmiento
    model.fit(x_train_poli,y_train)
    y_pred_pr = model.predict(x_test_poli)

    # calculate rmse and r2
    rmse = np.sqrt(mean_squared_error(y, y_new))
    r2 = r2_score(y, y_new)
    print('RMSE: ', rmse)
    print('R2: ', r2)


    # calculate rmse and r2
    rmse2 = np.sqrt(mean_squared_error(y_test, y_pred_pr))
    r22 = r2_score(y_test, y_pred_pr)
    print('RMSE: ', rmse2)
    print('R2: ', r22)



     # Datos Principales
    #plt.scatter(x,y_new,color='blue') #Datos con la Recta de Los datos Reales
    #plt.plot(x,model.predict(x_transform),color='black') #unon de la Recta

    plt.scatter(x,y_new,color="pink")

    plt.scatter(x_test,y_test,color ='green')
    plt.scatter(x_test,y_pred_pr,color='red',linewidth=3)
    plt.plot(x_test,y_pred_pr,color='black')
    plt.grid()
    #plt.plot(x_test,y_pred_pr,color='green',linewidth=3)
    titulo = 'Grado = {} RMSE = {} R2 = {}'.format(grado,round(rmse,2),round(r2,2))
    plt.title("Tendecia de COVID-19 en "+pais +"\n" + titulo, fontsize = 15)
    plt.xlabel("Fechas")
    plt.ylabel("Casos Confirmados")
    plt.show()

    return ""

def FuncLineal():
    pass