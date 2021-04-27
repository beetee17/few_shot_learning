import numpy as np
from Utils import buildModel

class Predictor(object):

    def __init__(self, name):
        self.n_correct = 0
        self.num_tests = 0
        self.name = name
        self.prediction = None
        self.use_training_set = False
        self.convert_2_rgb = False
        self.fsl = False
    
    def reset(self):
        self.n_correct = 0
        self.num_tests = 0
        self.prediction = None

    def get_name(self):
        return self.name

    def get_n_correct(self):
        return self.n_correct

    def get_num_tests(self):
        return self.num_tests

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, support_set, targets):
        # to be overridden by subclasses
        self.prediction = 0

    def update_score(self, support_set, targets, verbose):
        self.set_prediction(support_set, targets)
        if targets[self.get_prediction()] == 1:
            self.n_correct += 1
        self.num_tests += 1

        if verbose and self.get_num_tests() < 10:
            print('TEST {}'.format(self.get_num_tests()))
            print('{} GUESSED PAIR {}'.format(self.get_name(), self.get_prediction() + 1))
    
            if targets[self.get_prediction()] == 1:
                print("CORRECT")
            else:
                print("INCORRECT")
            print('\n')

    def calc_accuracy(self, N, K):
        acc = round((self.n_correct/self.num_tests) * 100, 2)
        print("{} Model achieved {}% accuracy on {} {}-way {}-shot tests".format(self.get_name(), acc, self.get_num_tests(), N, K))
        return acc
   

class FSL(Predictor):

    def __init__(self, model, name='Few Shot Learning', use_training_set=False, convert_2_rgb=False):
        super().__init__(name)
        self.model = model
        self.name = name
        self.probs = None
        self.use_training_set = use_training_set
        self.convert_2_rgb = convert_2_rgb
        self.fsl = True
        
    def get_probs(self):
        return np.round(self.probs, 4) 

    def set_prediction(self, support_set, targets):
        probs = self.model.predict([support_set[:,0], support_set[:,1]])
        self.probs = probs
        self.prediction = np.argmax(probs)
    
class Random(Predictor):

    def __init__(self, name = 'Random Guess'):
        super().__init__(name)
        self.name = name
       
   
    def set_prediction(self, support_set, targets):
        self.prediction = np.random.randint(0, len(support_set))
       


class Nearest_Neighbour(Predictor):

    def __init__(self, name ='Nearest Neighbour'):
        super().__init__(name)
        self.name = name
 
    def set_prediction(self, support_set, targets):
        # picks the class of image in the support set which has min L1 distance from query image
        min_dist = 10e9
        prediction = None
        for i in range(len(support_set)):
            pair = support_set[i]
            curr_dist = np.linalg.norm(pair[0].flatten() - pair[1].flatten(), ord=1)
            if curr_dist < min_dist:
                min_dist = curr_dist
                prediction = i
        self.prediction = prediction

def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax
    
class Model_Nearest_Neighbour(Predictor):

    def __init__(self, name='VGG16 Nearest Neighbour'):
        super().__init__(name)
        self.name = name
        self.model = buildModel.get_feature_extractor((200, 280, 3))
        self.predictions = []


    def set_prediction(self, support_set, targets):

        similarities = np.array([])

        for i in range(len(support_set)):
            pair = support_set[i]
            np.append(similarities, self.model.predict(pair))

        self.predictions = stable_softmax(similarities)
        self.prediction = np.argmax(predictions)



