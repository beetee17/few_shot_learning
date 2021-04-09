**To-Do (short-term)**
- literature review on current FSL algorithms(s), gather points of consideration for diff algos, base class selection
- detailed explanation of concepts/techniques involved in FSL (problem statement, deep learning, siamese networks/triplet loss)
- 

**To-Do (long-term)**
- narrow down on rare object to be classified (especially if FSL has better results with test cases similar to base classes)
- working python script implementing a FSL algorithm with test dataset
- specify end-product (web-service?)


**060421**

*Completed*
- confirm project topic as Few Shot Learning (FSL) for image classification on classes with limited available labelled samples 
- wrote simplified overview of problem statement and workings of FSL algorithm


*Learning Points*
- FSL leverages on CNN to train the model, hence it is not too dissimilar in terms of steps and techniques involved. However, it is still slightly more complex. Thus, a starting point would be to simply create a CNN to classify some readily available dataset before adapting the code to FSL.

**070421**

*Completed*
- more details added to understanding of FSL, explaining siamese network and CNNs
- downloaded omniglot dataset as a starting point
- wrote a jupyter notebook to preprocess dataset and build a prototype model


*Challenges*
- number of parameters are extremely large, computer takes too long to build the model and overheats (images are 105x105)
- need to evaluate next steps e.g. downsize images and/or reduce dimensionality with convolutions of larger strides and pooling layers

*Next to-dos*
- overcome aforementioned challenge and build a prototype model
- visualise its predictions to ensure the code worked as intended
- test on unseen classes in dataset

**080421**

*Completed*
- added functionality to notebook to save/load a model, run randomised n-way k-shot tests with relevant visualisations
- was able to build a model using 20 alphabets of dataset with muliple conv layers by reducing layers' dimensionalities (model had 90% accuracy on 100 out-of-sample 2-way-2-shot tests)

*Challenges*

- ~~a sufficiently accurate model would take a lot more power to build as current test model uses only 5 alphabets for training and just one convolutional layer in network~~
- ~~the model seen in https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d uses multiple conv and pooling layers, as well as 30 alphabets of dataset. It is stated that a T4000 GPU was used. Current machine only has intel HD630 GPU which is not utilised as Keras only supports Nvidia GPUs (not sure it would be much faster even if it could use the GPU)~~

*Next-to-dos*
- can try downsizing images to allow for increasing size of layers of model ( see if that produces even better model)
- make other models for validation purposes (e.g. random choice and k-nearest neighbour model)