import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model


model = load_model("crop_disease_model.h5", custom_objects={
                   'KerasLayer': hub.KerasLayer})
IMG_HEIGHT = 224
IMG_WIDTH = 224

img = cv2.imread("AppleCedarRust1.jpg")
resized_img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

labels_class = ['scab', 'black_rot', 'rust', 'healthy']

X = []
X.append(resized_img)

X_train = tf.expand_dims(X, axis=-1)
print(X_train.shape)

X = np.array(X_train)
print(X.shape)

predictions = model.predict(X)
print(predictions)

print("Predicted state: ", labels_class[np.argmax(predictions)])
