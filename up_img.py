import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
import pickle
from keras.models import load_model

UPLOAD_FOLDER = 'store_folder'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

new_model = load_model('model/age_model.h5')
with open('model/labels.pkl', 'rb') as handle:
    labels = pickle.load(handle)
with open('model/age_dict.pkl', 'rb') as handle:
    age_dict = pickle.load(handle)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            test_image = image.load_img(f'store_folder/{filename}', target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = new_model.predict(test_image)

            position_result = np.argmax(result)
            label_result = list(labels.keys())[position_result]
            prob_result = float(result[0][position_result])

            position_result = np.argmax(result)
            label_result = list(labels.keys())[position_result]
            prob_result = float(result[0][position_result])
            return f'La persona posee {age_dict[label_result]} con un {(prob_result*100):.2f}% de probabilidad!'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
