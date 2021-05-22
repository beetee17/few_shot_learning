# Problem statement
Few Shot Learning (FSL) is a sub-area of machine learning (ML) that can be used in Computer Vision  tasks, such as image classification. Unlike traditional ML algorithms that work on large datasets, FSL is designed to classify images (recognise objects) with few data samples. In essence, it is clear that highly accurate image classifying models can be built from training sets with thousands of positive and negative samples. However, not all classes are created equal. Some have few recorded instances for various reasons - a rare bone disease might have few X-rays available for study, or an enemy's military armoured vehicle might have few recorded sightings. 

These cases cannot be solved in the same way as conventional deep learning, and at its surface seems to be an almost impossible task for a machine to be able to recognise such classes any better than a random guess. 

# Intuition of FSL

It is obvious that we are, for example, able to recognise our parents' faces as we have seen them countless times from different angles and in different conditions. However, consider also that we are able to glance at someone's IC and tell immediately if the person in question is indeed the owner of his/her identifiaction. The human mind somehow has the ability to one-shot learn, let alone FSL. If deep learning attempts to emulate the workings of the human brain and the connection between its neurons to identify objects, maybe there exists some way to adapt the current deep learning techniques to objects with limited samples.

Returning to our example and taking a closer look, we realise that the brain is able to accurately recognise a face with a single positive sample because it is, in a sense, 'cheating'. It has likely seen millions of other faces, or even other objects, and trained itself to differentiate (or recognise the similarity between) any 2 objects. We can say that the brain has learnt to learn. This is the key finding that will allow us to train machines in FSL.

Meta learning, or learning to learn, means that a model learns the reason behind classification decisions, rather than a rote approach in the case of conventional deep learning. In this case, the reason our brains can tell if an unknown object belongs to some class is due to the similarity between the Query image (the person's actual face) and the Support Set (the person's IC).

A model of this example would be a *1-way 1-shot* classifier. However, a support set usually contains images from multiple classes. The model compares the query with each support image and outputs a similarity value for each image pair, usually between 0 and 1. The model's prediction would be the class of the support image that produced the highest similarity value when paired with the query. A more apt example would be a theft victim picking out the thief from a list of *n* suspects, with each suspect having *k* photos. This model would be a *n-way k-shot* classifier.


# How FSL Works
In traditional deep learning, the machine is tasked with recognising a preset list of classes (objects). Beyond that, it is relatively useless. For example, a model that classifies lions would be trained on a large dataset with images of lions (positive sample) and not lions (negative sample). The positive samples are labelled with 1, and 0 for negative samples. From there, the model uses an optimisation algorithm (e.g. gradient descent) to minimise some loss function that measures the error between the model's output is from the samples' label. From this very simplified explanation, we see that the model would only be able to accurately tell if an image was a lion or not given a sufficiently large training set. An image from any other class would simply result in a 0 output (i.e. a support set would be useless to the model). Moreover, if our dataset was small, the model would not be accurate in any way.

However, if we change the way we label our samples such that each pair of image would be a positive sample if they belong to the same class, and a negative sample otherwise, we are able to alter the model's goal to detecting the similarity of 2 images. This eliminates the need to train our model with samples of the rare class. We can use other classes which we have large datsets of to meta learn. Thus, when given an unseen query image from the rare class, we can use the one/few samples that we have and ask our model if they are the same object; this would yield a useful output.

In summary, FSL consists of the following:
1. Take the pair of images of a sample and iteratively pass them though a convolutional neural network for feature extraction
2. Take the absolute value of the difference between the 2 feature vectors
3. Process the resultant vector via dense layers to produce a scalar value
4. Apply the sigmoid function to obtain the model's prediction as a similarity value between 0 and 1
5. Compute the loss function by comparing the similarity value with the sample's label 
6. Calculate the gradients of the loss function with respect to the dense layers' parameters and perform gradient descent to update its parameters 
7. Further propagate the gradient to update the parameters of the neural network
8. Repeat using all training data
9. Test the model using a query image and a support set. Note that all the images involved do not belong to any of the classes found in the training data

# CNNs

A convolutional neural network (CNN) is a special kind of Feed Forward Neural Network that significantly reduces the number of parameters in a deep neural network with many units without losing too much in the quality of the model.

Due to the nature of images, where the important information within an image is concentrated in small regions, we can use the idea of convolutions (or filters) to scan the image with the aim of extracting the important features and reduce the dimensionality of our input with each convolution. 

A convolution, F, is simply a p x p matrix, that scans a similar p x p patch of the input image, and outputs a matrix (usually smaller than the input image depending on stride and padding) containng the dot products as it convolves from left to right, top to bottom till the end of the input image is reached. 

A convolutional layer consists of multiple convolutions, each with p x p parameters that can be trained during the backpropagation step to extract the 'best' features of an image.

Multiple convolutional layers may be used in a neural network, with the idea that each subsequent layer extracts increasingly finer/abstract features of an image, leading to greater accuracy of the model. If the CNN has one convolution layer following another convolution layer, then the subsequent layer l + 1 treats the output of the preceding layer l as a collection of size l image matrices (where size l is the number of filters in the previous layer), known as a volume. The size of that collection is called the volume’s depth. Each filter of layer l + 1 convolves the whole volume.

Pooling layers may be combined with convolutional layers to increase accuracy of th model while reducing the dimensionality of its input, resulting in quicker training. It works similar to a convolution, however, a fixed operator is applied to each patch, e.g. max or average, instead of a filter matrix. This means that the pooling layer has no trainable parameters.

# Siamese Network
In order to build our model to learn a similarity function, we have to consider pairs of images as the input instead of a single image in traditional image classification. This can be done using a siamese network. The training flow of the siamese network is as follows:
1. Input the pair of images of a sample, one after the other, into the same CNN
2. The outputs would be 2 feature vectors, f1 and f2, one for each image of the pair
3. Apply some distance function to the two vectors, such as taking the absolute value of f1 - f2, to obtain a single vector z
4. Pass z through a dense layer that outputs a scalar, and compute a similarity value by applying the sigmoid function
5. Calculate the loss between the sample label and the similarity value
6. Repeat for all samples in our dataset, updating the parameters of the CNN via backpropagation
7. Repeat for a number of epochs


# Omniglot Dataset

My first goal was to create a working architecture of a siamese network. The omniglot dataset was used to validate that the model was working as intended. It is a dataset consisting of 30 novel alphabets such as Arabic, Hebrew, Sanskrit, etc. Each alphabet contains around 20 characters each, and totals to over 600 classes. Each character has just 20 samples, which are 105 x 105 black and white images of handwritten drawings. The structure of the datset, as well as the standardisation of its samples allowed me to focus on creating the model, rather than spend too much time preprocessing the data. Thus, I chose to use this dataset as a starting point.

(go through omniglot.ipynb)
- minimal preprocessing (loading daaset, making sample pairs)
- architecture of siamese network (2 of same CNN feeding feature vectors to dense layers -> sigmoid)
- test against baselines (show plotted pdfs)
- implemented early stopping callback (show learning curve) + idea of fine tuning a pretrained model i.e transfer learning to further increase model accuracy
- show test against previous model 

# Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft)

After learning how to create a siamese network and implement few shot learning with the model, I applied the same workflow as the omniglot dataset to a military aircraft dataset. This dataset is a subset of the open source aircraft dataset known as FGVC aircraft dataset. I filtered out the commercial airline models from the dataset, leaving with 30 classes of aircraft which have some sort of miliatry usage. The 30 classes were split into 20 for training, 5 for validation and 5 for testing. Each class has 100 samples, which are coloured images of varying sizes at different angles. 

(show aircraft_prep.ipynb)
- loading of dataset
- splitting into train, test, val sets
- making positive and negative pairs

(show all_LCs.pdf)
- explain various measures to lower validation loss, including batch normalisation and dropout and varying batch sizes.
- settled on batch norm with batch size of 50 which produced best results with the given training set
- as for varying the architecture i.e. making a deeper model - explain that the increase in number of dense layers drastically increase training time due to number of nodes. Also tried to use a 'deeper' base network in the form of resnet50 instead of vgg16 which is 50 layers deep, but did not get as good results as vgg16.

# Batch Size

There is some literature to suggest that batch sizes affect generalisation ability of a deep neural network. The idea is that a relatively small batch size allows the model to in a sense 'jump' out of local minima due to the increased noise with each mini-batch. This optimal batch size needs to be found through experimentation, as too small a size may lead to poor convergence due to noise, while too large a size may cause convergence on local minima. 

# Batch Normalisation

Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.

# Dropout

Dropout is applied between two successive layers in a network. At each iteration a specified percentage of the connections (selected randomly), connecting the two layers, are dropped. This causes the subsequent layer rely on all of its connections to the previous layer.

# Making a simple web-service

After doing what I could to maximise the model's accuracy, I used FastAPI to create a simple web-service that would take user uploaded images to query the model and return a predicted class. I used Docker to containerise this web service.

(demo of web-service via Docker image)