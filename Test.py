import numpy as np
from keras.models import load_model

from keras.preprocessing.image import img_to_array


import cv2
from time import sleep

from keras.preprocessing import image

face_classifier = cv2.CascadeClassifier('/Users/nader/Desktop/Folders/Courses/Deep Learning/Practice/Project/haarcascade_frontalface_default.xml')
classifier =load_model('/Users/nader/Desktop/Folders/Courses/Deep Learning/Practice/Project/Emotion_little_vgg.h5')

class_labels = ['Cold','Warm','Neutral']
