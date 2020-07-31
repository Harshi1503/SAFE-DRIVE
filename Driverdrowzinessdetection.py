import pygame
import cv2   
import os   
from keras.models import load_model
import pygame
import numpy as np
from pygame import mixer
import time
mixer.init()
sound = mixer.Sound('alarm.wav')
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
lbl=['Close','Open']

model = load_model('driverDrowsinessModel.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)       
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(grayimg, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(grayimg)
    right_eye =  reye.detectMultiScale(grayimg)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye = r_eye / 255
        r_eye =  r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis = 0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0] == 1):
            lbl = 'Open' 
        if(rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h,x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0] == 1):
            lbl = 'Open'   
        if(lpred[0] == 0):
            lbl ='Closed'
        break
