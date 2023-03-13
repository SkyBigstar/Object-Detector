import tensorflow as tf
from tensorflow import keras
from keras import layers

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


m = keras.models.load_model('model.h5')

image_size=(216, 216)
batch_size=32

img=keras.preprocessing.image.load_img(
    "Images/free-to-use-sounds-Vkt3uDeDkdg-unsplash.jpg",target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
predictions = m.predict(img_array)
score = predictions[0]
print(
    "This image is x1 position %.2f and y1 position %.2f and x2 position %.2f and y2 position %.2f."
    % (score[0]*216,score[1]*216,score[2]*216,score[3]*216)
)



import cv2
img=cv2.imread('Images/free-to-use-sounds-Vkt3uDeDkdg-unsplash.jpg')
cv2.rectangle(img,(round(score[0]*216),round(score[1]*216)),(round(score[2]*216),round(score[3]*216)),(0, 255, 0))
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


