import os
import sys
from flask import Flask, render_template,request,url_for
from flask.json import jsonify
from werkzeug.utils import redirect, secure_filename 
sys.path.insert(1, './Analisis')
from analizararchivo import AnalizarArchivo
UPLOAD_FOLDER = './archs/'
ALLOWED_EXTENSIONS = set(['csv','xls','xlsx','json'])



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about') 
def about():
    return render_template('about.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['myfile']
      filename = secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      archivonuevo = filename
      dataframe = Analisis.LeerArchivo("./archs/"+archivonuevo)
      return dataframe.to_string()  #redirect(url_for('home'))
		

#-------------------------------------------------------------------------------------------
archivonuevo = ""
Analisis = AnalizarArchivo()


#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
