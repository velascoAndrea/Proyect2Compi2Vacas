import os
from flask import Flask, render_template,request,url_for
from flask.helpers import flash
from werkzeug.utils import redirect, secure_filename 

UPLOAD_FOLDER = './archs/'
ALLOWED_EXTENSIONS = set(['csv'])

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
      return redirect(url_for('home'))
		

if __name__ == '__main__':
    app.run(debug=True)
