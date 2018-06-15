import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class LSTMModel(object):
    '''
    LSTM model for approximated IQUCT
    One model for each action
    '''
    def __init__(self, input_dim, n_actions, nb_epoch):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.nb_epoch = nb_epoch # training epoch
        self.init_models()

    def init_models(self):
        '''
        Models initialization
        '''
        self.models = []
        for _ in range(self.n_actions):
            self.models.append(Sequential())
            self.models[-1].add(
                LSTM(
                    units=1,
                    batch_input_shape=(1, 1, self.input_dim), # (batchsize, nbtimesteps, inputdim)
                    stateful=True
                )
            )
            self.models[-1].add(Dense(1))
            self.models[-1].compile(loss='mean_squared_error', optimizer='adam')

    def reset(self):
        '''
        Reset the parameters.
        '''
        #TODO

    def update(self, data):
        '''
        Update the model wrt to the input data.
        Data points should have the form (a, x, y)
        '''
        # Separate the data
        X = [[] for i in range(self.n_actions)]
        y = [[] for i in range(self.n_actions)]
        for d in data:
            X[d[0]].append(d[1])
            y[d[0]].append(d[2])
        X = np.array(X)
        y = np.array(y)
        # Train the models
        for i in range(self.n_actions):
            Xi = X[i].reshape(X[i].shape[0], 1, X[i].shape[1])
            yi = y[i].reshape(X[i].shape[0])
            print('Xi shape = {}'.format(Xi.shape))#TRM
            print('yi shape = {}'.format(yi.shape))#TRM
            for _ in range(self.nb_epoch):
                self.models[i].fit(Xi, y[i], epochs=1, batch_size=Xi.shape[0], verbose=0, shuffle=False)
                self.models[i].reset_states()
        print('---FINISH---')
        exit()#TRM

    def prediction_at(self, a, x):
        '''
        Compute a prediction at the given input
        '''
        return 0#TRM
        #TODO
