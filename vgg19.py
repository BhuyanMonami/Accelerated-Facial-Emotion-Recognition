import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from os import listdir
import argparse
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

# Count the number of images
input_folder = "jaffedbase"
count = 0
for root_dir, cur_dir, files in os.walk(input_folder):
    count += len(files)
print('file count:', count)
# # Divide images in training and testing sets
output_folder = "JAFFE_new"
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.9, .1), group_prefix=None)
# Check if splitting has been done correctly

for root, subdirs, files in os.walk(output_folder):
	for dirs in subdirs:
		print(dirs, len(os.listdir(os.path.join(root, dirs))))
# Load images using the image data generator
#data augmentation
train_dir = os.path.join(output_folder, "train")
train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
print("No of training samples:"+str(train_samples))
val_dir = os.path.join(output_folder, "val")
val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])
print("No of validation samples:"+str(val_samples))
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=str, required=True)
parser.add_argument('--val_batch_size', type=str, required=True)
args = parser.parse_args()
train_batch_size = int(args.train_batch_size)
val_batch_size = int(args.val_batch_size)
print(val_samples//val_batch_size)
# Load pre-trained model on ImageNet dataset
model = VGG19(weights='imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3))

model.summary()

# Add few more layers to train
x = keras.layers.Flatten()(model.layers[-1].output)
x = keras.layers.Dense(512, activation="relu")(x)
out = keras.layers.Dense(7, activation="softmax")(x)
# Set the layers that can be trained
for layer in model.layers[:-1]:
    layer.trainable = False
# Summarize the model
model = keras.models.Model(inputs=model.input, outputs=out)
model.summary()
print('This is the number of trainable weights:', len(model.trainable_weights))
# compiling
model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-5),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
)

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=train_batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),  # use the same pixel h * w target size for all images processed
    batch_size=val_batch_size,  # Batch size defines the number of samples that will be propagated through the network
    class_mode='categorical'
)
# Set stop to 5 epochs to prevent overfitting
callback = EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=5
)
# fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=(train_samples//train_batch_size)+1, #steps_per-epoch = total no of training images//batch size
    epochs=100,
    validation_data=validation_generator,
    validation_steps=(val_samples//val_batch_size),
    callbacks=[callback]
)
# Plot training performance of model
train_acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_acc, label='Training acc')
plt.plot(val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(train_loss, label='Training losses')
plt.plot(val_loss, label='Validation losses')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Save VGG19 results
model.save("trained_model_VGG19.h5")















