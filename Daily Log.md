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


**130421**
*To-Do*
- research on current OCR space (techniques, rare languages) etc
- Counter terrorism applications
- Make model detect unseen classes i.e if query image is not in support set (how would the model know? Maybe if its below some threshold in max predicted probabilty) -> useful for model to be able to detect if class is not in database

**140421**
*Completed*
- OCR space seems to be saturated even with more difficult and rare languages that have counter-terrorism applications such as Arabic
- For now, goal remains to be to classify novel military vehicles/weapon systems/etc

*Challenges*
- Read up on fine-tuning as a means to improve accuracy of FSL model, however, am unsure at what exactly it entails 
- This video explained the whole process: https://youtu.be/U6uFOIURcD0, however, I'm unsure if my understanding of it is correct, especially the part on training the model's parameters. This is because he states that the weights are initialise to be mean feature vectors of the support set. However, if each few shot task is done on different support sets, the mean vectors are changing from task to task, and therefore how could they be updated (since they are reset with each query)
- The only way I can think of to make the parameters trainable, is if our support set is our entire dataset, excluding the query images which we select beforehand (would it make a difference if we include them? My thinking is that if we include them it is in a sense 'cheating' as our model gets some part of the answer already)

- In my understanding, we first have a pre-trained neural network that is capable of extracting features from images. This neural network is trained on a large base dataset of non-novel classes. I also infer from this that if our base dataset is very different from our novel classes, the features extracted may be less relevant to our goal, especially at the deeper layers of the neural network -> thus, if we do use such a pretrained network, our fine tuning stage could backpropagate its gradients to the last few layers of neural network. However, if our network was pretrained on a base dataset similar to novel classes, we can at most simply backpropagate our gradient to the last layer of the network, if at all. The neural network may be a straightforward conv neural network that predicts labels of images, or it could be a siamese network architecture that predicts similarity scores. This is less important because when fine tuning, we strip the network of its activation layer, as we only care about the feature vector that is output from the network. 
- Now comes the actual fine tuning. We have our novel dataset that our network was not trained on. This datset contains few samples of each class. Let's say we have 20 images per class with a total of 10 classes. We choose how many images per class to use in our fine-tuning, let's say we choose 5 images each, for a total of 5x10 query samples.
- We exclude these 50 chosen queries and pass the rest of the labelled images to the feature extractor (i.e. our pretrained network) to output a feature vector for the class. Since there are >1 sample (20-2 = 18) of the novel class, we take the mean of its feeature vectors. Then, we normalize it. Thus, for each of the 10 classes in the novel dataset, we have a corresponding mean feature vector as our base.
- now, we can make various n-way-k-shot tasks to fine tune our model. Since we chose 2 images per class to use as queries, we haven 20 queries in total, and 18 images per class in the support set. This means we have 20 10-way-18-shot tasks.
- For each of the queries, we now pass the query image through the feature extractor and obtain a vector. This vector is paired with each of the 10 mean feature vecotrs of the support set, and a distance metric is computed (e.g. absolute diff, euclidean distance, cosine similarity, etc). This results in a vector of similarities between our query (of unknown class to the model) and the various classes in our novel database. Now, we apply a softmax to this similarity value to output the model's probability vector that our query image belongs to each of the classes. 
- Cross entropy loss can be used as our loss function, and backpropagate its gradients to update the model parameters (which were initialised as the mean feature vectors). We may also have a bias parameter, which we can init as 0
- Since we may have small suppport sets and/or query samples, we must use a regularizer to prevent overfitting  (entropy regularization was suggested for this)
  

*To-Do*
- Implement the above. The current model may be used as the pretrained-network. Thus, what needs to be done is:
- Split the 'test' dataset into a 'fine-tuning' and 'actual_test' dataset (2:1 split?) 
- In the fine-tuning set, exclude a number of images from each character, e.g. 5 per character.
- Our support set consists of the rest of the images in the fine-tuning datsset
- Compute the mean feature vectors for each class in the support-set by passing the images into our pretrained model (strip the lambda and sigmoid layer)
- Create a second model and initialise its parameters to be the mean vectors
- Iterate through each query image, passing them through the pretrained model to get a feature vector; normalize it
- Compute a similarity value with each class in the support set and apply softmax
- Compute total loss and backpropagate


**150421**
*Completed*
- All of the above was completed, however it turned out to be a dead end (i.e too dumb to understand the video)
- Now moving to fine tuning a pretrained model using siamese network architecture


*To-do*
- Proceed to clean up + preprocess military dataset
- Use simple fine-tuned model approach and apply to the dataset


**160421**
*Completed*
 - modify nearest neighbour to be abs diff in output of feature vector from pretrained model (instead of just abs diff of the images themselves)
- graph accuracy of the fine-tuned model and see if it obtains better results than previous model (from scratch)
 - 
*To-do*
-
- plot new graph
- cont. training current pretrained model as it seems Out Of Sample performance is still equal to training set performance
- Clean up military dataset