import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import sys
from os import listdir
import random
from random import shuffle
from math import floor
import cv2
import shutil
from shutil import copyfile
import pathlib
from pathlib import Path
import splitfolders

import numpy as np
import pandas as pd

import tensorflow
from tensorflow import keras
from keras import models, layers, losses, optimizers, metrics
from keras.models import Sequential, load_model, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19

import matplotlib.pyplot as plt

model1 = keras.models.load_model("trained_model_VGG19.h5")
print(model1.summary())
# Extract weights, biases and dimensions of each layer after training
#print("Weights and biases of the layers after training the model: \n")
for layer in model1.layers:
    if len(layer.get_weights()) == 0:
        continue
    print(layer.name)
    print("Weights")
    print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
    # print("Bias")
    print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')
shape_list = []
for layer in model1.layers:
    if len(layer.get_weights()) == 0:
        continue
    # For weights
    weights_bias_list = []
    weights = layer.get_weights()[0]
    weight_T = np.transpose(weights)
    for val in weight_T:
        for v in val:
            weights_bias_list.extend(v.T.flatten())

    # For bias
    weights_bias_list.extend(layer.get_weights()[1])
    with open('model_weights/' + layer.name + '.txt', 'w') as fw:
        for s in weights_bias_list:
            fw.write(str(s) + '\n')
    # For shapes/dimensions
    shape_list.extend(list(layer.get_weights()[0].shape))

with open('dimensions.txt', 'w') as fw:
    for shape in shape_list:
        fw.write(str(shape) + '\n')

# Validation
output_folder = "JAFFE_new"
val_dir = os.path.join(output_folder, "val")
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),  # use the same pixel h * w target size for all images processed
    batch_size=1,  # Batch size defines the number of samples that will be propagated through the network
    class_mode='categorical'
)
# Print the predicted output from model
labels = {
    0 : "ANGRY",
    1 : "DISGUST",
    2 : "FEAR",
    3 : "HAPPY",
    4 : "NEUTRAL",
    5 : "SAD",
    6 : "SURPRISE"
}

for _ in range(24):
    data = next(validation_generator)
    x = data[0]
    print(data[1][0])
    y = labels[np.nonzero(data[1][0])[0][0]]
    print(y)
    pred =model1.predict(x)
    plt.imshow(x.reshape(224, 224, 3))
    if (pred[0][0] > 0.5):
        legend=labels[0]
    if (pred[0][1] > 0.5):
        legend=labels[1]
    if (pred[0][2] > 0.5):
        legend=labels[2]
    if (pred[0][3] > 0.5):
        legend=labels[3]
    if (pred[0][4] > 0.5):
        legend=labels[4]
    if (pred[0][5] > 0.5):
        legend=labels[5]
    if (pred[0][6] > 0.5):
        legend=labels[6]
    print(legend)
    plt.title("Original:" + y + ",Predicted:" + legend)
    plt.show()
    print()
