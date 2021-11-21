!pip install kaggle

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# Then move kaggle.json into the folder where the API expects to find it.
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions list

!kaggle datasets download "jehanbhathena/weather-dataset"

!unzip "weather-dataset.zip"

import sklearn

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

classnames = os.listdir("/content/dataset")
classnames

!wget https://raw.githubusercontent.com/dilshans2k/TensorFlow-Deep-Learning/main/extras/helper_functions.py
import helper_functions as hf

def view_multiclass_random_images(path):
  _classnames = os.listdir(path)

  fig = plt.figure(1,figsize = (10,10))
  i=0
  # Helper Link - https://stackoverflow.com/a/41385215
  Tot = len(_classnames)
  Cols = int(Tot**0.5)
  Rows = Tot // Cols
  Rows += Tot%Cols
  Position = range(1,Tot+1)



  for folder in _classnames:
    fig.add_subplot(Rows,Cols,Position[i])
    i=i+1
    random_img = hf.random.choice(os.listdir(path + "/" + folder))
    random_img_path = path + "/" + folder + "/" + random_img
    img = plt.imread(random_img_path)
    plt.imshow(img)
    plt.title(folder)
    plt.axis(False)

view_multiclass_random_images("/content/dataset")

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

label = []
path = []

for folder in os.listdir("/content/dataset"):
  for file in os.listdir("/content/dataset/"+folder):
    path.append("/content/dataset/"+folder+"/"+file)
    label.append(folder)

df = pd.DataFrame(columns = ['path','label'])
df['path'] = path
df['label'] = label
df

X_train, X_test = train_test_split(df,test_size = 0.2,random_state = 42)

X_train.shape, X_test.shape

"""Trying without data augmentation

"""

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  #  shear_range = 0.2,
                                  #  width_shift_range = 0.2,
                                  #  height_shift_range=0.2,
                                  #  horizontal_flip = True,
                                  #  zoom_range=0.2,
                                  
                                   validation_split = 0.3
                                   )
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_data = train_datagen.flow_from_dataframe(dataframe = X_train, 
                                               x_col = 'path',
                                               y_col = 'label',
                                               class_mode = 'categorical',
                                               target_size = (224,224),
                                               subset = 'training',
                                               batch_size = 32)
valid_data = train_datagen.flow_from_dataframe(dataframe = X_train,
                                               x_col = 'path',
                                               y_col = 'label',
                                               class_mode = 'categorical',
                                               target_size = (224,224),
                                               subset = 'validation',
                                               batch_size = 32)

test_data = test_datagen.flow_from_dataframe(dataframe = X_test,
                                             x_col = 'path',
                                             y_col = 'label',
                                             class_mode = 'categorical',
                                             target_size = (224,224),
                                             batch_size = 32,
                                             shuffle = False)

test_data[0][0].shape

valid_data.class_indices

base_model_1 = MobileNetV2(include_top = False)
base_model_1.trainable = False

input = hf.tf.keras.layers.Input(shape = (224,224,3))
x = base_model_1(input,training = False)
x = tf.keras.layers.GlobalAveragePooling2D(name = 'GAP2D')(x)
x=hf.Dense(128, activation='relu')(x)
x=hf.tf.keras.layers.Dropout(0.2)(x)
x=hf.Dense(64, activation='relu')(x)
x = hf.tf.keras.layers.Dropout(0.2)(x)
output = hf.Dense(len(classnames),activation='softmax')(x)

model_1 = hf.tf.keras.Model(input,output)

model_1.summary()

model_1.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

history_1 = model_1.fit(train_data,
                        epochs = 10,
                        validation_data = valid_data)

model_1.evaluate(test_data)

model_1.save("weather_mobilenetv2.h5")

