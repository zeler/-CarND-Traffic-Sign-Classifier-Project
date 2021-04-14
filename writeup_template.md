# **Traffic Sign Recognition** 

## Summary 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/dataset_samples.jpg "Dataset samples"
[image2]: ./img/label_distribution.jpg "Label distribution"
[image3]: ./img/grayscale.jpg "Grayscale images"
[image4]: ./img/smote.jpg "SMOTE"
[image5]: ./img/smote_dist.jpg "SMOTE distribution"
[image6]: ./img/test1.jpg "Test image"
[image7]: ./img/test2.jpg "Test image"
[image8]: ./img/test3.jpg "Test image"
[image9]: ./img/test4.jpg "Test image"
[image10]: ./img/test5.jpg "Test image"
[image11]: ./img/visualized_layers.jpg "Visualized layers"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zeler/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the inbuilt numpy functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

The dataset is highly unbalanced, more specifically, we have following label counts per class in the training sample (class is equal to the lavel count in the array):

```python
import numpy as np
print(np.bincount(y_train))

[ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920
  690  540  360  990 1080  180  300  270  330  450  240 1350  540  210
  480  240  390  690  210  599  360 1080  330  180 1860  270  300  210
  210]
```

#### 2. Include an exploratory visualization of the dataset.

For an exploratory visualization of the dataset, I started with plotting some examples of the traffic sign images along with their labels, so I could get grasp on the type of images I'm going to deal with:

![Dataset samples][image1]

Apart from that, I also plotted the histogram of label distribution along the training sample:

![Label distribution][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscales as this allows the network to fix on other features than color of the images, which could significantly change in different lighting conditions. 

![Grayscale images][image3]

Next, I normalized the image data as it was required by the project rubric. Normalization helps to improve the training speed and reduces the chances of getting stuck in local optima.

As seen in previous chapters, the dataset is highly unbalanced. To cope with this, I decided to generate augmented data using Synthetic Minority Over-sampling Technique (SMOTE). SMOTE is an oversampling technique for generating synthetic samples from the minority classes. The result is the well-balanced training set, which I later used to train the classifier. More information about this technique can be found [here](https://medium.com/swlh/how-to-use-smote-for-dealing-with-imbalanced-image-dataset-for-solving-classification-problems-3aba7d2b9cad).

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_train_rs, y_train)
```

Here is an example of augmented images for class 25:

![SMOTE][image4]

The class distribution for the dataset now looked like this:

![SMOTE Distribution][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Batch normalization   |                                               |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x12 	|
| Batch normalization   |                                               |
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 6x6x30  	|
| Batch normalization   |                                               |
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 5x5x30   				|
| Flatten               | outputs 750                                   |
| Fully connected		| outputs 240       							|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| outputs 168       							|
| RELU					|												|
| Dropout				| 												|
| Fully connected       | outputs 43                                    | 
| Softmax				|           									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, as it can be used instead of classical SGD and, at the same time, has many benefits - like computational efficiency, being appropriate for problems with noisy data and the intuitivnes of hyper-parameters. 

I trained the network during 150 epochs with batch size of 8 and learning rate 0.00008. Since I chose to run the algorit for quite a few epochs, I implemented simple early-stop procedure, where I saved the model with best accurracy on validation set. I decided to use relatively small batch size and learning rate as it clearly helped with training in later epochs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 95.3% 
* test set accuracy of 92%

To find a solution, I chose to use na iterative approach when designing the network. I started with the Lenet network built during the curicullum class, however, I quickly realized the network proably won't be sufficient, given a big difference in number of output classes in the class solution and the problem I was trying to solve. Because of that, I decided to increase the number of fully connected nodes in the network. 

As I was expecting this could end up with overfitting, I also implemented dropout layer to help the network with generalization. After having multiple cycles of training of the netowrk from scratch, it seemed the network has troubles with getting the accurracy on validation set to some sensible numbers, so I decided to use another convolutional layer to help with feature extraction as well. 

To help with the training, I also added batch normalization as suggested in web articles, as this has helps in convolutional layers with better generalization (similar to droput, which doesn't make sense to implement in convolutional layers). On top of these techniques, I also implemented L2 regularization to further prevent overfitting and promote generalization. 

Running the network like this, I did some extra experiments with hyperparameters and figured out that the network has better results with slower learing speeds and smaller batch sizes (at the expense of training time). I was able to get results very close to 92% accurracy at validation layer, so I finally decided to augment the images using SMOTE, which proved to be effective and I got to about 95% (or more) accuracy on the validation set.  
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. More specifically, I randomly picked these from the IJCNN 2011 competition dataset. 

![Test image][image6] 

The first image might be difficult to classify because there is a part of other sign visible as well. Apart from that, this image seems to be of good quality overall and should be easy to classify.

![Test image][image7] 

The second image pose a bigger challenge, since there are 3 signs in the image. Also the image has low contrast iand is dark overall.

![Test image][image8] 

The third image might be difficult to clasify becuase of a strong edge created by horizont in the picture. Also the resolution is low and it's difficult to guess what symbol is on the sign. 

![Test image][image9] 

The fourth image may be a challenge because the number on the sign is difficult to read - it could be easily mistaken for a different symbol.

![Test image][image10]

The last image has good contrast, but again, the symbols are difficult to see. Also there is a half of a sign on top of the image with similar symbols in it. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image	Label	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0			      		| 0   											| 
| 12     				| 12 											|
| 30					| 30											|
| 5	      				| 5								 				|
| 9						| 9				      							|


The model was able to correctly classify all 5 traffic signs, which gives an accuracy of 100%. When compared with test set accurracy, which is 92%, it seems like we are in the same range, however, there are only few examples on which I tested the model, hence the high accurracy. In real-world scenario, I would expect much lower accurracy. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model seems to be very certain in its predictions most of the time.

For the first image, the probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9978     			| 0         									| 
| .0021   				| 1 											|
| .00003				| 4												|
| .00002	   			| 38							 				|
| .000008			    | 6				      							|

For the second image the probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9648     			| 12         									| 
| .0077   				| 35											|
| .0075					| 9												|
| .0034	   				| 41							 				|
| .0032				    | 40			      							|

For the third image the probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9945     			| 30         									| 
| .0012   				| 25											|
| .0010					| 11											|
| .0010	   				| 23							 				|
| .0004				    | 28			      							|

For the fourth image the probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9817     			| 5         									| 
| .0064   				| 10											|
| .0053					| 3 											|
| .0076	   				| 2 							 				|
| .0004				    | 7      		      							|

For the fifth image, the probabilities were as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9996     			| 9         									| 
| .0001   				| 10											|
| .00009				| 16 											|
| .00002   				| 12							 				|
| .00001			    | 19      		      							|

Overall, there is very high confidence about the classified image and the network isn't in almost any doubts in this case. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I tried to visualize the layers for one of the training images. It seems the convolutions are getting excited with overall shapes in the first layer (circular shape of the sign) and later are focusing in more and more details of a given convolutional layers. This comes hand-in-hand with how we transform the spatial information from "wide" (the original image) to deep (the convoluted image).

![Visualized layers][image11]

