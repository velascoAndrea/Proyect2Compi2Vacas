import os
import sys
from flask import Flask, render_template,request,url_for
from flask.json import jsonify
from werkzeug.utils import redirect, secure_filename 
sys.path.insert(1, './Analisis')
from analizararchivo import AnalizarArchivo
import json as json


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
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      archivonuevo = filename
      Data = Analisis.LeerArchivo("./archs/"+archivonuevo)
      return jsonify(Data.to_string(),filename)  #redirect(url_for('home'))
		


@app.route('/NombreColumnas',methods = ['POST', 'GET'])
def ImpresionConsola():
    if request.method == 'POST':
        envio = request.form
        nombreArchivo = envio['filename']
        Data = Analisis.LeerArchivo("./archs/"+nombreArchivo)
        print(Data.columns.values.tolist())
        return jsonify(Data.columns.values.tolist())



@app.route('/Reporte1EnvioParametros',methods = ['POST', 'GET'])
def Reporte1EnvioParametros():
    if request.method == 'POST':
        envio = request.form
        #en la posicion 0 esta el Pais
        parametros = json.loads(envio['columnas'])
        print(envio)
        print(type(parametros))
        print(parametros[1])
        #print(envio['filename'])
        return ""



#-------------------------------------------------------------------------------------------
archivonuevo = "" #nombre del Archivo Leido
Analisis = AnalizarArchivo()
Data = None #DataFramedeLosDatos 



#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
