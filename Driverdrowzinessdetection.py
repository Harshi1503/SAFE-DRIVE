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
