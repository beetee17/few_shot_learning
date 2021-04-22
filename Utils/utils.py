import matplotlib.pyplot as plt
import numpy as np

def show_pairs(num_pairs, pairs, labels, raw_labels=np.array([]), random=False):

    for i in range(num_pairs):
    
        fig = plt.figure()
        rand_i = i
        if random:
            rand_i = np.random.randint(0, len(labels))
        
        if len(raw_labels) > 0:
            fig.suptitle('{}\n{}                             {}'.format(labels[rand_i], raw_labels[rand_i][0], raw_labels[rand_i][1]))

        elif not raw_labels:
            fig.suptitle('{}\n'.format(labels[rand_i]))

        plt.subplot(2,2,1)
        plt.imshow(pairs[rand_i][0])
        plt.subplot(2,2,2)
        plt.imshow(pairs[rand_i][1])
        


        
        