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

**090421**

*Completed*
- added 2 baseline models to compare against FSL model, specifically random guess and nearest neighbor model
- added 20 way 1 shot testing function for rigorous test of FSL model (i.e an alphabet is chosen, and one character is made the query image while 1 image from each of the 20 characters within the alphabet make up the support set)

*Next-to-dos*
- research on more techniques to improve accuracy of model
- brainstorm on which rare object to choose for end-product (ideally it is relevant to DSTA, has related open source datasets for training)

*Challenges*
- prototyping a model using 60% of dataset with multiple conv layers takes a couple of hours


**120421**

*Completed*
- completed 20-way 1-shot teasting function. Some alphabets had less than 20 characters. The current function outputs a task up to 20-way 1-shot (if alphabet has 17 chars -> 17-way 1-shot task is produced. If alphabet had 30 chars -> 20-way 1-shot task is produced with random selection of chars)
- added early callback when trainng model using loss on validation testing between epochs (if loss on validation set increases, stop training and save the previous model)
- Have not prototyped new model using this method as it takes quite long
- Added function to graph accuracy of different models/baselines on many N-way 1-shot tests for different Ns

*Next to-dos*
- See 090421
- Create model using train, validation split and early callback
- Compare this model against current 
- Repeat using k-fold cross validation and statified k-fold?