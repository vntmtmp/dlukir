import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_mysqldb import MySQL
# from flask_mysqldb import MySQLdb, MySQLdb
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import cv2
import math


UPLOAD_FOLDER = 'D:/01test/flask app/assets/images/upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

filepath = 'D:/01test/models/model200 (1).h5' 
model = load_model(filepath)
# print(model)

# print("Model Loaded Successfully")

@app.route('/', methods=['GET', 'POST'])
def root():
   return render_template('home.html')

@app.route('/home.html', methods=['GET', 'POST'])
def home():
   return render_template('home.html')

@app.route('/index.html', methods=['GET', 'POST'])
def index():
   return render_template('index.html')   

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/login.html')
def login():
   return render_template('login.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/Pa_lulun_pao.html')
def Pa_lulun_pao():
   return render_template('Pa_lulun_pao.html')

@app.route('/Pa_somba.html')
def Pa_somba():
   return render_template('Pa_somba.html')

@app.route('/Pa_tangke_lumu.html')
def Pa_tangke_lumu():
   return render_template('Pa_tangke_lumu.html')

@app.route('/Pa_tumuru.html')
def Pa_tumuru():
   return render_template('Pa_tumuru.html')


def pred_ukir(ukiran):
  test_image = load_img(ukiran, target_size = (224, 224)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D
  
  result = model.predict(test_image) # prediksi
  print('@@ Raw result = ', result)
  hasil = result.tolist()
  print(hasil)
  hasil2 = str(hasil[0])
  print(hasil2)
  hasil6 = hasil2.split()
  hasil4 = hasil6[0]
  print(hasil6)
  print(hasil4)
  hasil5 = hasil4.replace(",","")
  hasil8 = hasil5.replace("[","")
  hasil7 = float(hasil8)
  print('hasil 7 adalah',hasil7)
  
  if hasil7 <0.1:
   print('akurasi=100%')
  elif hasil7 <0.2:
   print('akurasi=10%') 
  elif hasil7 <0.3:
   print('akurasi=20%')
  elif hasil7 >0.3:
   print('akurasi=0%')

  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return "Pa Lulun Pao",'Pa_lulun_pao.html'

  elif pred==1:
      return "Pa Somba",'Pa_somba.html'
        
  elif pred==2:
      return "Pa Tangke Lumu",'Pa_tangke_lumu.html'
        
  elif pred==3:
      return "Pa Tumuru",'Pa_tumuru.html'
  else:
      # return "Tidak Terdeteksi, tidak_terdeteksi.html"
      print('tidak terdeteksi')
# Create flask instance
@app.route('/prediksi', methods = ['GET','POST'])
def prediksi():
   if request.method == 'POST':
      print("Load Model Berhasil")
      file = request.files['file'] # fet input
      print(file)
      filename = "upload.jpg"    
      file_path1 = os.path.join('D:/01test/flask app/assets/images/upload/', filename)
      file.save(file_path1)
      print(file_path1)
      print("@@ Input posted = ", filename)
      #canny
      img2 = cv2.imread('D:/01test/flask app/assets/images/upload/upload.jpg')  # Read image
      # Setting parameter values
      t_lower = 50  # Lower Threshold
      t_upper = 150  # Upper threshold      
      # Applying the Canny Edge filter
      edge = cv2.Canny(img2, t_lower, t_upper)
      # simpan citra canny
      cv2.imwrite('D:/01test/flask app/assets/images/upload/canny.jpg', edge)

      # filename2 = "canny.jpg"  
      # file_path2 = os.path.join('D:/01test/flask app/assets/images/upload/', filename2)
      # print(file_path2)
      # print("@@ Predicting class......")
      # ddept=cv2.CV_16S  
      # img2 = cv2.imread('D:/01test/flask app/assets/images/upload/upload.jpg')
      # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
      # x = cv2.Sobel(gray, ddept, 1,0, ksize=3, scale=1)
      # y = cv2.Sobel(gray, ddept, 0,1, ksize=3, scale=1)
      # absx= cv2.convertScaleAbs(x)
      # absy = cv2.convertScaleAbs(y)
      # edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
      # cv2.imwrite('D:/01test/flask app/assets/images/upload/canny.jpg', edge)

      filename2 = "canny.jpg"  
      file_path2 = os.path.join('D:/01test/flask app/assets/images/upload/', filename2)
      print(file_path2)
      print("@@ Predicting class......")
      #--------------------------------#      

      pred, output_page = pred_ukir(file_path1)           
      return render_template(output_page, pred_output = pred, user_image = file_path1)

# For local system & cloud
# if __name__ == "__main__":
#    app.run(threaded=False,port=8080) 
if __name__ == '__main__':
    host = os.getenv('IP','0.0.0.0')
    port = int(os.getenv('PORT',5000))
    app.secret_key = os.urandom(24)
    app.run(host=host,port=port)