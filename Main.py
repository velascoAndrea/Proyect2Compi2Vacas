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
from pandas.io.json import json_normalize

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


@app.route('/reportes') 
def reportes():
    return render_template('reporte.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['myfile']
      #print(f.content_type,"ARCHIVO")
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      archivonuevo = filename

      if(f.content_type=="text/csv"):
        pass
        #Data = Analisis.LeerArchivo("./archs/"+archivonuevo)

      return jsonify("Data",filename)  #redirect(url_for('home'))
		


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
        if(extension[len(extension)-1] =='xls' or extension[len(extension)-1] =='xlsx'):
            Data = Analisis.LeerArchivoExcel("./archs/"+nombreArchivo)

        if(extension[len(extension)-1] =='json'):
            Data = Analisis.LeerArchivoJson("./archs/"+nombreArchivo)
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

        if(extension[len(extension)-1] =='xls' or extension[len(extension)-1] =='xlsx'):
            Data = Analisis.LeerArchivoExcel("./archs/"+nombreArchivo)

        if(extension[len(extension)-1] =='json'):
            Data = Analisis.LeerArchivoJson("./archs/"+nombreArchivo)     
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
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep1 = rep.AnalizarRep1("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
        #print(AnalisisRep1[1],"RETORNO IMAGen")
        an = '<img src=\'data:image/png;base64,{}\'>'.format(AnalisisRep1[0])

    #En la Posicion 0 va la imagen
    #En la Pocicion 1 el tipo de Reporte
    #En la Posicion 2 la imagen en base 64
    #En la POsicion 3 el Pais
    #En la Posicion 4 el R2
    return  jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep1[0]),AnalisisRep1[1],AnalisisRep1[0],AnalisisRep1[3],AnalisisRep1[5],AnalisisRep1[6])


@app.route('/AnalisisReporte2',methods = ['POST', 'GET'])
def AnalisisReporte2():
    print("Prediccion de infectados en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        fecha = envio['fecha']
        print(fecha,'FEHCAAAAAA')
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep2 = rep.AnalizarRep2("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2],fecha)
        prediccion = AnalisisRep2[7].tolist()
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep2[0]),AnalisisRep2[1],AnalisisRep2[0],AnalisisRep2[3],AnalisisRep2[5],AnalisisRep2[6],prediccion)


@app.route('/AnalisisReporte4',methods = ['POST', 'GET'])
def AnalisisReporte4():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep4 = rep.AnalizarRep4("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep4[0]),AnalisisRep4[1],AnalisisRep4[0],AnalisisRep4[3],AnalisisRep4[5],AnalisisRep4[6])



@app.route('/AnalisisReporte5',methods = ['POST', 'GET'])
def AnalisisReporte5():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep5 = rep.AnalizarRep5("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep5[0]),AnalisisRep5[1],AnalisisRep5[0],AnalisisRep5[3],AnalisisRep5[5],AnalisisRep5[6])

@app.route('/AnalisisReporte9',methods = ['POST', 'GET'])
def AnalisisReporte9():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep9 = rep.AnalizarRep9("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep9[0]),AnalisisRep9[1],AnalisisRep9[0],AnalisisRep9[3],AnalisisRep9[5],AnalisisRep9[6])


@app.route('/AnalisisReporte15',methods = ['POST', 'GET'])
def AnalisisReporte15():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep15 = rep.AnalizarRep15("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep15[0]),AnalisisRep15[1],AnalisisRep15[0],AnalisisRep15[3],AnalisisRep15[5],AnalisisRep15[6])

@app.route('/AnalisisReporte14',methods = ['POST', 'GET'])
def AnalisisReporte14():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep14 = rep.AnalizarRep14("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep14[0]),AnalisisRep14[1],AnalisisRep14[0],AnalisisRep14[3],AnalisisRep14[5],AnalisisRep14[6])


@app.route('/AnalisisReporte6',methods = ['POST', 'GET'])
def AnalisisReporte6():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep6 = rep.AnalizarRep6("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2])
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep6[0]),AnalisisRep6[1],AnalisisRep6[0],AnalisisRep6[3],AnalisisRep6[5],AnalisisRep6[6])


@app.route('/AnalisisReporte10',methods = ['POST', 'GET'])
def AnalisisReporte10():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        pais2 = envio['pais2']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep10 = rep.AnalizarRep10("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2],pais2)
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep10[0]),AnalisisRep10[1],AnalisisRep10[0],AnalisisRep10[3],AnalisisRep10[5],AnalisisRep10[6],AnalisisRep10[7],AnalisisRep10[8])


@app.route('/AnalisisReporte12',methods = ['POST', 'GET'])
def AnalisisReporte12():
    print("Prediccion de Mortalidad en un pais")
    if request.method == 'POST':
        envio = request.form
        parametros = json.loads(envio['columnas'])
        #print(parametros)
        nombreArchivo = envio['filename']
        pais = envio['pais']
        pais2 = envio['pais2']
        extension = nombreArchivo.split('.')
    #if(extension[len(extension)-1] =='csv'):
        AnalisisRep12 = rep.AnalizarRep12("./archs/"+nombreArchivo,pais,parametros[0],parametros[1],parametros[2],pais2)
    return jsonify('<img class=\'img-thumbnail\' src=\'data:image/png;base64,{}\'>'.format(AnalisisRep12[0]),AnalisisRep12[1],AnalisisRep12[0],AnalisisRep12[3],AnalisisRep12[5],AnalisisRep12[6],AnalisisRep12[7],AnalisisRep12[8])




#-------------------------------------------------------------------------------------------
archivonuevo = "" #nombre del Archivo Leido
Analisis = AnalizarArchivo()
rep = Reportes()
Data = None #DataFramedeLosDatos 

#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
