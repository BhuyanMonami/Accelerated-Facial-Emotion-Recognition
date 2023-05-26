# Accelerated-Facial-Emotion-Recognition
This project tries to implement a CUDA-based architecture to improve performance and thus reduce 
latency in running inference during the feedforward propagation of a VGG19 model architecture-based 
Convolutional Neural Network(CNN). Here, a popular dataset called the Japanese female facial 
expression (JAFFE) dataset consisting of 213 images and 7 universal facial expressions is used for 
training the model using transfer learning and then performing inference on both Python(using Keras 
and TensorFlow backend) and CUDA and the objective is to make a comparison between the both.

# Problem Statement:
One drawback of CNN is that the training requires very large datasets to converge  to their global optima and the huge networks take a long time, even for inferencing. It is seen that most of the execution time of a convolutional layer is spent performing convolutions.5 Companies like Nvidia, Intel, Google, Qualcomm, AMD, and Microsoft have developed AI chips for either training a new network or inference on an already trained network, using the weights and biases values from training performed in the past. Super quick response time from these AI chips are going to be used for very critical applications such as self-driving cars. Thus, there is a need to speed up the inference to make these chips run faster. This creates the need to find ways to optimize the operations going on in these layers. To do so, we can leverage the decades of research done to optimize matrix-matrix multiplication. While the cuBLAS library has CUDA GEMM API and Intel MKL has special optimized CPU GEMM the hardware AI chips don‟t have these libraries built into them. Hence the programmers need to perform all operations themselves and optimize the same.
This project aims to parallelize and thus speed up the feedforward of the VGG-19 architecture using a CUDA-based parallel computing architecture. This approach thus aims to efficiently exploit the GPU execution resources and in-core memories. Transfer learning for FEA has been chosen to achieve the same. Low latency and optimized inference can find many benefits in FEA devices used in critical industries like health and manufacturing. 

# Solution description:
1. The Japanese female facial expression (JAFFE) dataset consisting of 213 images in which 10 female subjects are expressing their faces in 7 universal emotions like anger, disgust, fear, happiness, sadness, surprise, and neutral is chosen. 11 This dataset is chosen to see how a small dataset responds to training the model. The size of the dataset is 256 x 256 x 3. 
2. A transfer learning method based on previous work done by Akhand et al., has been adopted to initially check how the model performs using Keras and TensorFlow backend 
within PyCharm IDE. The algorithm deploys the VGG19 architecture, freezing the pretrained layers and adding two dense layers as our classifier along with a softmax layer.
3. The input images are resized into 224 x 224 (as required by the VGG19 model), rescaled by a scaling factor of x1.1, and horizontally flipped. Adam optimizer with a learning rate of 0.0005 is chosen. The images are randomly split into training and validation folders in the ratio of 90:10. A callback function with Early Stopping is also used while training the model to prevent overfitting.
4. The model is trained and the accuracy and validation metrics graphs are plotted.
5. The prediction is then made and the inference time is noted.
6. The updated weights and biases of the trained model are extracted in separate text files for each convolutional layer. The output shapes (or dimensions) of the trained model are also extracted and stored in a separate text file. (These files are uploaded to the repository). 
7.  These parameters are then utilized for parallelizing the feedforward of the VGG19 architecture using CUDA using the following approach:
The convolutions are converted into general matrix-to-matrix multiplications(GEMM) and parallelism is used at each layer in the forward propagation to improve memory efficiency and achieve a lower execution time. A function called “im_to_col” is employed to first convert the convolution operation into a simple GEMM operation. For any convolutional layer, filter-sized image patches(3x3) are extracted, with the given stride length and converted into columns. It‟s also made sure that the channels come one below the other in a column. To maintain the operation of convolution as expected, the filters are converted into rows, which get multiplied by the columns of im_to_col. The channels in the filter are hence placed (channel-wise) one after the other in the row. The fully connected layer directly can be implemented with a GEMM operation, so im_to_col is not required in this case.

# Deliverables:
The complete implementation of the code is as follows:
- Download the JAFFE dataset either from http://www.kasrl.org/jaffe_download.html or from the Project repository sub-directory named **"jaffedbase"**.
- Save the JAFFE dataset in the same directory as the codes.
- Run the file vgg19.py using the following command on the terminal:
**python vgg19.py --train_batch_size 32 --val_batch_size 24**
(The code will randomly split the dataset into two sub-directories namely ‘train’ and ‘val’, with 
189 and 24 samples respectively).
- Save the accuracy and loss metrics curve.
- Save the trained model as: trained_model_VGG19.h5
- Run the file emotion_recognition.py on the terminal using the following command:
**python emotion_recognition.py**
- The emotion_recognition.py will generate the prediction images and the model weights for each layer as a text file and another text file with the shapes. Save the model weights as a subdirectory called **"model_weights"** in the same directory as your code. Likewise, save the dimensions file (**dimensions.txt**). 
- The code runs inference for one image to check the veracity and as a preliminary step towards hopefully a more robust algorithm in the future.
- Compile the emotion_recognition.cu file using the following command on the terminal:
**nvcc -c emotion_recognition.cu**
- The compiler creates an emotion_recognition.o object file. Then implement the inference using the following command on the terminal:
**nvcc -o emotion_recognition emotion_recognition.o**

