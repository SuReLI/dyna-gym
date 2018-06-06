import numpy as np
from lwpr import LWPR

class LWPRForIQUCT(object):
    '''
    LWPR model for approximated IQUCT
    '''
    def __init__(self, n_in):
        self.n_in = n_in # Input dimensions
        self.n_out = 1 # Output dimensions

        # Model
        self.model = LWPR(self.n_in,self.n_out)
        # Model parameters #TODO
        self.model.init_D = 20 * np.eye(self.n_in)
        self.model.update_D = True
        self.model.init_alpha = 40 * np.eye(self.n_in)
        self.model.meta = False

        self.n_training_repetition = 1 #TODO

    def reset():
        '''
        Reset the parameters.
        '''
        #TODO

    def update(self, data):
        '''
        Update the model wrt to the input data.
        Data points should have the form (s, t, a, Q) where Q is the predicted value
        '''
        n = len(data)
        for k in range(self.n_training_repetition):
            ind = np.random.permutation(n)
            for i in range(n):
                st = np.array(data[ind[i]][0] + data[ind[i]][1])
                q = np.array(data[ind[i]][3], ndmin=2)
                #TODO separate models for each action
                yp = self.model.update(st, q)

    def prediction_at(self, s, t, a):
        '''
        Compute a prediction at the given input
        '''
        st = np.array(s + t)
        return self.model.predict(st)
