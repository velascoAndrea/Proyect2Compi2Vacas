from math import degrees
import numpy as np
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
from datetime import date, datetime
from sklearn.metrics import mean_squared_error, r2_score
import base64
from io import BytesIO
from flask.json import jsonify

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
        fechas = fechas.apply(datetime.toordinal)
        #creo un nuevo DataFrame Para agruparlo por Fecha
        ds =pd.DataFrame()
        ds[independiente] = fechas
        ds[dependiente] = dataFrameFiltrado[dependiente] 
        ds = ds.groupby(by=independiente).sum()
        #print(ds.keys())
        print(ds.index)
       
        #Fecha X Variable Independiente
        x = ds.index.values
        #Numero de Casos Y Dependiente
        y = ds[dependiente].values

        #Tengo que reordenar para que funcione
        x = x.reshape(len(x),1)
        
        print(x.shape,"DATO X")
        print(y.shape,"DATO Y")

        #--------------------------------------ANALISIS----------------------------------------------------------
        
        figpoli =  FuncPoli(x,y,pais,3)
        
        #print(figpoli,'AQUIII')
        return figpoli


    def AnalizarRep2(self,nombreArchivo,pais,columnaPais,dependiente,independiente):
        
        dataFrame = pd.read_csv(nombreArchivo);
        is_country = dataFrame[columnaPais] == pais
        #print(is_country)
        dataFrameFiltrado = dataFrame[is_country]
        #pasar el formato a fecha
        fechas = pd.to_datetime(dataFrameFiltrado[independiente])
        fechas = fechas.apply(datetime.toordinal)
        #creo un nuevo DataFrame Para agruparlo por Fecha
        ds =pd.DataFrame()
        ds[independiente] = fechas
        ds[dependiente] = dataFrameFiltrado[dependiente] 
        ds = ds.groupby(by=independiente).sum()
        #print(ds.keys())
        print(ds.index)
       
        #Fecha X Variable Independiente
        x = ds.index.values
        #Numero de Casos Y Dependiente
        y = ds[dependiente].values

        #Tengo que reordenar para que funcione
        x = x.reshape(len(x),1)
        
        print(x.shape,"DATO X")
        print(y.shape,"DATO Y")

        #--------------------------------------ANALISIS----------------------------------------------------------
        
        figpoli =  FuncPoliPrediccion(x,y,pais,2)
        #print(figpoli,'AQUIII')
        return figpoli

    
def  FuncPoli(x,y,pais,gradoAsi):
    #Datos de Entrenamiento
    print(x.shape,y.shape)
    fig = plt.figure()
   
    x_train,x_test, y_train, y_test = train_test_split(x,y)
    grado = gradoAsi
    polynomial_reg = PolynomialFeatures(degree = grado)

    x_transform = polynomial_reg.fit_transform(x)  #.astype(float) # La mera X
    x_train_poli = polynomial_reg.fit_transform(x_train) # La X de entrenamiento
    x_test_poli = polynomial_reg.fit_transform(x_test) # La X test
    
    print(x_transform.shape,y.shape)
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
    aa = datetime.fromordinal(737792)
    print("AA",aa)
    print(type(x),"AQUI")
    #print(x)
    i = []
    j = []
    for n in x:
        i.append(datetime.fromordinal(int(n)))

    for n in x_test:
        j.append(datetime.fromordinal(int(n)))        
    #print(i)
    #i = datetime.fromordinal(int(x[0,0]))
    
    

    plt.scatter(i,y)
    plt.scatter(i,y_new,color="pink")
    plt.plot(i,y_new,color="grey")

    #plt.scatter(j,y_test,color ='green')
    #plt.scatter(j,y_pred_pr,color='red',linewidth=3)
    #plt.plot(j,y_pred_pr,color='black')
    #plt.grid()
    #plt.plot(x_test,y_pred_pr,color='green',linewidth=3)
    titulo = 'Grado = {} RMSE = {} R2 = {}'.format(grado,round(rmse,2),round(r2,2))
    plt.title("Tendecia de COVID-19 en "+pais +"\n" + titulo, fontsize = 15)
    plt.xlabel("Fechas")
    plt.ylabel("Casos Confirmados")
    #plt.show()
    fig.set_figheight(10)
    fig.set_figwidth(16)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return [encoded, pais, rmse, rmse2]


def  FuncPoliPrediccion(x,y,pais,gradoAsi):
    #Datos de Entrenamiento
    print(x.shape,y.shape)
    fig = plt.figure()
   
    x_train,x_test, y_train, y_test = train_test_split(x,y)
    grado = gradoAsi
    polynomial_reg = PolynomialFeatures(degree = grado)

    x_transform = polynomial_reg.fit_transform(x)  #.astype(float) # La mera X
    x_train_poli = polynomial_reg.fit_transform(x_train) # La X de entrenamiento
    x_test_poli = polynomial_reg.fit_transform(x_test) # La X test
    
    print(x_transform.shape,y.shape)
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
    aa = datetime.fromordinal(737792)
    print("AA",aa)
    print(type(x),"AQUI")
    #print(x)
    i = []
    j = []
    for n in x:
        i.append(datetime.fromordinal(int(n)))

    for n in x_test:
        j.append(datetime.fromordinal(int(n)))        
    #print(i)
    #i = datetime.fromordinal(int(x[0,0]))
    
    

    plt.scatter(i,y)
    plt.scatter(i,y_new,color="pink")
    plt.plot(i,y_new,color="grey")

    plt.scatter(j,y_test,color ='green')
    plt.scatter(j,y_pred_pr,color='red',linewidth=3)
    plt.plot(j,y_pred_pr,color='black')
    plt.grid()
   
    titulo = 'Grado = {} RMSE = {} R2 = {}'.format(grado,round(rmse2,2),round(r22,2))
    plt.title("Prediccion de Infectados en "+pais +"\n" + titulo, fontsize = 15)
    plt.xlabel("Fechas")
    plt.ylabel("Casos Confirmados")
    #plt.show()
    fig.set_figheight(10)
    fig.set_figwidth(16)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return [encoded, pais, rmse, rmse2]    





#-------------------------------------REVISAR----------------------------------------------

def Lineal(x,y,pais):
    print(type(x),"TIPO X")
    print(type(y),"TIPO Y")
    y = y.reshape(len(y),1)
    x = pd.to_datetime(x)
    print(x,"ESTA ES LA X")
    print(y,"ESTA ES LA y")
    print(x.shape,y.shape)
    fig = plt.figure()
    plt.scatter(x,y)
    x_train,x_test,y_train, y_test = train_test_split(x,y)

    #Defino el algoritmo a utilizar
    model = linear_model.LinearRegression()
    #Entreno el Modelo
    model.fit(x,y)
    #Realizo la prediccion
    print(x.shape,y.shape)
    #y_new = model.predict(x)

    #Entreno el Modelo
    #model.fit(x_train,y_train)
    #Realizo la prediccion
    #print(x_train.shape,y_train.shape)
    #y_pre = model.predict([10])

    #plt.scatter(x,y_new,color="pink")
    #plt.plot(x,y_new,color="grey")

    plt.show()
    return ''

