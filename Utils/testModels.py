import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def get_test(X_test, Y_test, N, K):
    
    # get num_tests amount of queries and support sets (n way k shot)
    # final output is a list of lists of tuples. Each inner list is a test containing tuple pairs in the form: [(Q, s1), (Q, s2),..., (Q, sk)]
    # where Q is the query img and s1 to sk are k imgs of the support set
    # second output is a list of targets(answers)
    # init lists of queries and support sets

    support_set = []
    targets= np.zeros(N*K)
    test_classes = list(set(Y_test))
   
    # choose random index
    i = np.random.randint(0, len(X_test))

    # get query img and corresponding label
    query = X_test[i]
    label = Y_test[i]
    
    # get 1 * K pairs that is same class
    for k in range(K):
        pos_j = [j for j, class_ in enumerate(Y_test) if class_ == label]
        j = np.random.choice(pos_j)

        pair = [query, X_test[j]]
        targets[k] = 1
        support_set.append(pair)

    # get n-1 * K pairs that are of diff class
    test_classes.remove(label)

    support_classes = []
    for n in range(N-1):

        random_class = np.random.choice(test_classes)
        support_classes.append(random_class)        
    
        for k in range(K):
            
            pos_j = [j for j, class_ in enumerate(Y_test) if class_ == random_class]
            j = np.random.choice(pos_j)
            pair = [query, X_test[j]]
            support_set.append(pair)
            
        test_classes.remove(random_class)

   
    # shuffle support set and add to all support sets
    support_set, targets = shuffle(support_set, targets)

    return np.array(support_set), np.array(targets)
    

def test_models(models, X_test, Y_test, num_tests, N, K, verbose = 0):
    """function to test any or all of our models that belong to the same Model class on the same test set of
    N way K shot cases over num_test tasks. Use verbose=1 setting to output prediciton data and create relevant visualisations!"""

    # limit number of graphs to be plotted for visualisation
    MAX_GRAPHS = 20
    CURR_GRAPHS = 0

    if verbose:
        print("Evaluating models on {} random {} way {} shot learning tasks ... \n".format(num_tests, N, K))


    for num in range(num_tests):
        
        support_set, targets = get_test(X_test, Y_test, N, K)

        for model in models:
            
            copy_support_set = support_set[:]
            copy_targets = targets[:]

            if model.use_training_set:
                # 'Test' this model on the training set (for comparison purposes)
                
                copy_support_set, copy_targets = get_test(X_train, Y_train, N, K)
        
            if model.convert_2_rgb:
                # Convert 1 channel images to 3 channel for prediction
                copy_support_set3=np.zeros(copy_support_set.shape +(3,))

                for i in range(len(copy_support_set)):
                    for j in range(2):
                        img = np.stack((copy_support_set[i][j],)*3, axis=-1)
                        copy_support_set3[i][j] = img

                copy_support_set = copy_support_set3

            model.update_score(copy_support_set, copy_targets, verbose)

            
           

            if model.fsl and verbose and CURR_GRAPHS <= MAX_GRAPHS:

                for i in range(len(support_set)):
        
                    fig = plt.figure()
                    ax1 = fig.add_subplot(221)
                    ax1.imshow(support_set[i][0], cmap='gray')
                    ax2 = fig.add_subplot(222)
                    ax2.imshow(support_set[i][1], cmap='gray')
                    fig.suptitle('TEST {}, PAIR {}\n{}: {}, actual: {}'.format(num + 1, i+1, model.name, model.get_probs()[i], targets[i]))

                    CURR_GRAPHS += 1

    accuracy = {}
    for model in models:
        accuracy.update({model.name : model.calc_accuracy(N, K)})
    return accuracy

def get_accuracy(models, X_test, Y_test, N, num_tests):
    '''tests various model(s)/baseline(s) using num_tests amount of n-way 1-shot tasks (n belongs to N)
    and output their mean accuracies over all tests for each n in N'''

    # init 
    accuracies = {name : [] for name in [model.name for model in models]}
    accuracies.update({'range_' : N})
    
    for n in N:

        if n != 1:
            print('testing {}-shot 1-way...'.format(n))

        if n == 1:
            for k, v in accuracies.items():
                if k != 'range_':
                    accuracies[k].append(100)
            continue

        accuracy = test_models(models, X_test, Y_test, num_tests, n, K=1, verbose = 0)

        # reset attributes of models
        for model in models:
            model.reset()

        for k, v in accuracy.items():
            accuracies[k].append(v)

    return accuracies

def plot_accuracy(accuracies, save=None):
    
    plt.figure()

    N = accuracies['range_']
    for k, v in accuracies.items():

        # ignore the range entry (they are the x values)
        if k != 'range_':
            plt.plot(N, v, label=k)

    # Set the axes labels and fix x axise intervals
    plt.xlabel('# of Classes')
    plt.xticks(N)
    plt.xlim(N[0], N[-1]+1)
    
    plt.ylabel('Model Accuracy (%)')

    # Set title of the current axes.
    plt.title('Prediction Accuracy vs # of Classes in 1-Shot Support Set')

    # show legend on the plot
    plt.legend()

    if save:
        plt.savefig(save)
        print("Graph was saved as {}!".format(save))

    # Display the figure.
    plt.show()
