import numpy as np
import cv2
import os
from flask import Flask,flash, request, jsonify, render_template  
app = Flask(__name__,template_folder='Template')
@app.route("/")
def about():
    return render_template('index.html')   
@app.route("/inner")
def ab1():
    return render_template('inner.html')
@app.route('/predict',methods=['POST'])    
def home():
    data1= request.form['name']
    data2= request.form['roll']
    data3= request.form['course']
    data4= request.form['branch']
    data5= request.form['father']
    data6= request.form['id']
    data7= request.form['num']
    data8= request.form['mobile']
    countt=2
    #data9= request.form['rc']
    file = request.files['dl'] 
    file_name = str(countt)+".jpg"
    file.save(file_name)
    counttt=3
    file = request.files['card']
    file_name = str(counttt)+".jpg"
    file.save(file_name)
    cnt=5
    file = request.files['pic'] 
    file_name = str(cnt)+".jpg"
    file.save(file_name)
    image = cv2.imread(file_name)
    scale_percent = 60 # percent of original size
    width = 100#int(image.shape[1] * scale_percent / 140)
    height = 85#int(image.shape[0] * scale_percent / 250)
    dim = (width, height)
    image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    path =r'C:\Users\hriti\Desktop\innotech\Techie\Techie\static\assets\img\5.jpg'
    cv2.imwrite(path,image)
    ## Load the cascade  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')  
      
    # Read the input image 
    fil=r'C:\Users\hriti\Desktop\innotech\Techie\2.jpg'
    fill=r'C:\Users\hriti\Desktop\innotech\Techie\3.jpg'
    img = cv2.imread(fil)      
    img1= cv2.imread(fill) 
    
    # Convert into grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  
      
    # Detect faces  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4) 
    
    # Draw rectangle around the faces  and slice them
         
    count =0
    for(x,y,w,h) in faces:
        face=img[y:y+h,x:x+w]     
        cv2.imwrite(str(count)+'.jpg',face)
        count+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(250,0,0),2)  
    
    
    count1 =0
    for(x,y,w,h) in faces1:
        faces1=img1[y:y+h,x:x+w]     
        cv2.imwrite(str(count)+'.jpg',faces1)
        count1+=1
        cv2.rectangle(img1,(x,y),(x+w,y+h),(250,0,0),2)
    
    
    count=0
    # Display the output  
    img3=cv2.imread(str(count)+'.jpg')
    img4=cv2.imread(str(count1)+'.jpg')
    
    img3 = cv2.resize(img3, dsize=(200, 200))
    img4 = cv2.resize(img4, dsize=(200, 200))
    
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
    def mse (img3,img4):
        h, w=img3.shape
        diff=cv2.subtract(img3,img4)
        err=np.sum(diff**2)
        mse=err/(float(h*w))
        return mse, diff
    
    error,diff=mse(img3,img4)
    if(error<17):
        return render_template('true.html',name=data1,roll=data2,course=data3,branch=data4,id=data6,nump=data7)
    else:
        error = 'FACES DO NOT MATCH. Please try again!'
        return render_template('inner.html',error=error)    

if __name__ == '__main__':
    app.run(port=5000, debug=True)      
         