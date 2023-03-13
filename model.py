import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers

from keras.models import Sequential
import pathlib
import pandas as pd
from PIL import Image
from PIL.ImageDraw import Draw


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 4 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


width = 216
height = 216
num_classes = 2
classes = ["Circle", "No-Circle"]



TRAINING_CSV_FILE = 'Data/training_data.csv'
TRAINING_IMAGE_DIR = 'Images/Training'

training_image_records = pd.read_csv(TRAINING_CSV_FILE)

train_image_path = os.path.join(os.getcwd(), TRAINING_IMAGE_DIR)

train_images = []
train_targets = []
train_labels = []

for index, row in training_image_records.iterrows():
    (filename, width, height, class_name, xmin, ymin, xmax, ymax) = row

    train_image_fullpath = os.path.join(train_image_path, filename)
    train_img = keras.preprocessing.image.load_img(train_image_fullpath, target_size=(height, width))
    train_img_arr = keras.preprocessing.image.img_to_array(train_img)

    xmin = round(xmin / width, 2)
    ymin = round(ymin / height, 2)
    xmax = round(xmax / width, 2)
    ymax = round(ymax / height, 2)

    train_images.append(train_img_arr)
    train_targets.append((xmin, ymin, xmax, ymax))
    train_labels.append(classes.index(class_name))




VALIDATION_CSV_FILE = 'Data/validation_data.csv'
VALIDATION_IMAGE_DIR = 'Images/Validation'

validation_image_records = pd.read_csv(VALIDATION_CSV_FILE)

validation_image_path = os.path.join(os.getcwd(), VALIDATION_IMAGE_DIR)

validation_images = []
validation_targets = []
validation_labels = []


for index, row in validation_image_records.iterrows():
    (filename, width, height, class_name, xmin, ymin, xmax, ymax) = row

    validation_image_fullpath = os.path.join(validation_image_path, filename)
    validation_img = keras.preprocessing.image.load_img(validation_image_fullpath, target_size=(height, width))
    validation_img_arr = keras.preprocessing.image.img_to_array(validation_img)

    xmin = round(xmin / width, 2)
    ymin = round(ymin / height, 2)
    xmax = round(xmax / width, 2)
    ymax = round(ymax / height, 2)

    validation_images.append(validation_img_arr)
    validation_targets.append((xmin, ymin, xmax, ymax))
    validation_labels.append(classes.index(class_name))


train_images = np.array(train_images)
train_targets = np.array(train_targets)
train_labels = np.array(train_labels)
validation_images = np.array(validation_images)
validation_targets = np.array(validation_targets)
validation_labels = np.array(validation_labels)

#create the common input layer
input_shape = (height, width, 3)
input_layer = keras.Input(input_shape)
#create the base layers

base_layers = layers.Rescaling(1./255, name='bl_1')(input_layer)
base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
base_layers = layers.Flatten(name='bl_8')(base_layers)
#create the localiser branch
locator_branch = layers.Dense(128, activation='relu', name='bb_1')(base_layers)
locator_branch = layers.Dense(64, activation='relu', name='bb_2')(locator_branch)
locator_branch = layers.Dense(32, activation='relu', name='bb_3')(locator_branch)
locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)
model = keras.Model(input_layer, outputs=[locator_branch])
losses = {"bb_head": tf.keras.losses.MSE}
model.compile(loss=losses, optimizer='Adam', metrics=['accuracy'])
history = model.fit(train_images, train_targets,
             validation_data=(validation_images, validation_targets),
             batch_size=4,
             epochs=50,
             shuffle=True,
             verbose=1)
model.save('model.h5')
















