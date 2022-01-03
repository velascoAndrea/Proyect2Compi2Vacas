import os
import sys
from flask import Flask, render_template,request,url_for
from flask.json import jsonify
from werkzeug.utils import redirect, secure_filename 
import matplotlib.pyplot as plt
sys.path.insert(1, './Analisis')
from analizararchivo import AnalizarArchivo
from reportes import Reportes
import json as json
import numpy as np



UPLOAD_FOLDER = './archs/'
ALLOWED_EXTENSIONS = set(['csv','xls','xlsx','json'])



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about') 
def about():
    return render_template('parametrizacion.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['myfile']
      #print(f.content_type,"ARCHIVO")
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      archivonuevo = filename

      if(f.content_type=="text/csv"):
        Data = Analisis.LeerArchivo("./archs/"+archivonuevo)

      return jsonify(Data.to_string(),filename)  #redirect(url_for('home'))
		


@app.route('/NombreColumnas',methods = ['POST', 'GET'])
def ImpresionConsola():
    if request.method == 'POST':
        envio = request.form
        nombreArchivo = envio['filename']

        extension = nombreArchivo.split('.')
        print("Extension",extension[len(extension)-1])

        if(extension[len(extension)-1] =='csv'):
            Data = Analisis.LeerArchivo("./archs/"+nombreArchivo)
            #print(Data.columns.values.tolist())

        return jsonify(Data.columns.values.tolist())



@app.route('/Reporte1EnvioParametros',methods = ['POST', 'GET'])
def Reporte1EnvioParametros():
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        nombreArchivo = envio['filename']
        extension = nombreArchivo.split('.')
        print("Extension",extension[len(extension)-1])

        if(extension[len(extension)-1] =='csv'):
            Data = Analisis.LeerArchivo("./archs/"+nombreArchivo)
        #en la posicion 0 esta el Pais
            #print(Data[parametros[0]].dropna().drop_duplicates().values)

        #print(type(Data[parametros[0]].to_string()))
        #print(envio)
        #print(type(parametros))
        #print(parametros[0])
        #print(envio['filename'])

        #Quito los valores nulos y los Duplicado para obtener la Lista de Paises 
        return  jsonify(np.array(Data[parametros[0]].dropna().drop_duplicates().values).tolist())



@app.route('/AnalisisReporte1',methods = ['POST', 'GET'])
def AnalisisReporte1():
    #
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    if(extension[len(extension)-1] =='csv'):
        AnalisisRep1 = rep.AnalizarRep1("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
        #print(AnalisisRep1)
        
    return ""    


#-------------------------------------------------------------------------------------------
archivonuevo = "" #nombre del Archivo Leido
Analisis = AnalizarArchivo()
rep = Reportes()
Data = None #DataFramedeLosDatos 



#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
