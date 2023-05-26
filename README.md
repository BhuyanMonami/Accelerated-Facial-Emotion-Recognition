# Accelerated-Facial-Emotion-Recognition
This project tries to implement a CUDA-based architecture to improve performance and thus reduce 
latency in running inference during the feedforward propagation of a VGG19 model architecture-based 
Convolutional Neural Network(CNN). Here, a popular dataset called the Japanese female facial 
expression (JAFFE) dataset consisting of 213 images and 7 universal facial expressions is used for 
training the model using transfer learning and then performing inference on both Python(using Keras 
and TensorFlow backend) and CUDA and the objective is to make a comparison between the both.
