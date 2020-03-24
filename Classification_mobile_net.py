from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D

from keras.models import Sequential,Model 
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization


# NetMobile created to play with pictures in 200, 200 dimensions
img_cols,img_rows = 200,200

NetMobile = NetMobile(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))
