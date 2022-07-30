# ai07 Image Classification of Concretes With or Without Cracks using Convolutional Neural Network
 
## 1. Objective
The objective is to construct a convolutional neural network that able to classify the images dataset of concretes with or without the crack.Dataset is divided into two category such as negative(without crack) and positive(with crack).Besides,each category will provide 20000 images.

[Concrete dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2) this link will provide the dataset.

## 2. IDE and Framework
Juypter notebook is the main IDE.The main frameworks used in this project are Numpy,Tensorflow Keras and Matplotlib.

## 3. Methodology
Transfer Learning will be the main methodology in this project.The model was pre-trained from large dataset typically for classification task.The pre-trained model can be applied to a specific given task.For more detail please refer to this official [documentation](https://www.tensorflow.org/tutorials/images/transfer_learning).

# 3.1 Data Pipeline
Data annotation of concrete dataset was loaded.Firstly, data was split into train-validation-test set with a ratio images of 28000:9600:2400.Besides,the dimension of all images in the dataset were reshaped snd color channel was converted to 224x224 with a RGB channels(224,224,3).Next abundant dataset was given, data agumentation will not be applied in this project. 


# 3.2 Model
Pre-trained model of feature extractor MobileNet V3 small was chosen in this project due to the huge dataset provided.Hence small architecture of MobileNet was selected to reduce the training time.

The rules of image shape in MobileNet V3 is 224x224 with 3 color channels.Preprocessing layer for input of MobileNet V3 is a placeholder.Next,the input of the MobileNet V3 is a float tensor with a pixel in the range of [0-255].All the weight parameter will not updated during the training process since the layer of all pre-trained model were freezed.Apart from that,abundant dataset was provided the dropping rate of drop layer in MobileNet V3 will be increase to 0.3 to prevent overfitting.


Lastly, the global average pooling and dropout layer were implemented to vectorize the feature map and to prevent overfitting.Then,the signal will be fed into the output softmax regression layer to classify the images.

The simplified version layers of the model was shown in the figure below.
![Model](https://user-images.githubusercontent.com/109932205/181167345-cf037ba6-98dd-4edf-9c50-d309e4554a5d.png)


Abundant images was provided and pre-trained model was applied.Therefore, 5 epochs with 32 batch size was sufficient to train the model accurately as shown in table below.
|             | Training | Validation |
| ----------- | -------- | ---------- |
| Loss        | 0.0061   | 0.0057     |
| Accuracy(%) | 99.8     | 99.83      |

The training results were shown in the figure below.

![Process](https://user-images.githubusercontent.com/109932205/181167605-64d7d1cb-1030-46dd-8819-6892678609a8.png)




## 4. Results
The model test with tested data and result was shown in the figure below.

![Result1](https://user-images.githubusercontent.com/109932205/181174590-73cb0afd-7cb7-4780-aeb7-b3516db6847c.png)

![Result2](https://user-images.githubusercontent.com/109932205/181174642-db63353a-0fdf-4a75-8418-f3338d35bea9.png)





A couple of actual result images and predictions of the model images were shown in the picture below.
![image](https://user-images.githubusercontent.com/109932205/181175609-1717bbe2-0b6f-47f0-a513-9aa9967b5fdc.png)








**The model able to attain a classification accuracy of 100% for test result.**


