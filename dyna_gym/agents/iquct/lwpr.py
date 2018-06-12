import numpy as np
from lwpr import LWPR

class LWPRModel(object):
    '''
    LWPR model for approximated IQUCT
    One LWPR model for each action.
    Realized mapping is (s,t) -> Q for each a
    '''
    def __init__(self, state_dim, n_actions, n_training_repetition):
        self.n_in = state_dim + 1 # Input dimensions (time concatenate)
        self.n_out = 1 # Output dimensions
        self.n_actions = n_actions
        self.n_training_repetition = n_training_repetition
        self.init_models()

    def init_models(self):
        '''
        Model initialization.
        Set the parameters of the model to their initial values.
        Careful tuning must be done in this function.
        '''
        self.models = []
        for _ in range(self.n_actions):
            self.models.append(LWPR(self.n_in, self.n_out))
            self.models[-1].init_D = 100000 * np.eye(self.n_in)
            self.models[-1].update_D = True
            self.models[-1].init_alpha = 50 * np.eye(self.n_in)
            self.models[-1].meta = False

    def reset(self):
        '''
        Reset the parameters.
        '''
        self.init_models()

    def update(self, data):
        '''
        Update the model wrt to the input data.
        Data points should have the form (s, t, a, Q) where Q is the predicted value
        '''
        n = len(data)
        for k in range(self.n_training_repetition):
            ind = np.random.permutation(n)
            for i in range(n):
                st = np.concatenate(((data[ind[i]][0], np.array(data[ind[i]][1], ndmin=1))))
                q = np.array(data[ind[i]][3], ndmin=1)
                yp = self.models[data[ind[i]][2]].update(st, q)
        #self.plot_pred(data)

    def prediction_at(self, s, t, a):
        '''
        Compute a prediction at the given input
        '''
        st = np.concatenate((s, np.array(t, ndmin=1)))
        return self.models[a].predict(st)

    def plpred(self, X, Y, a):
        mesh = []
        for y in Y:
            l = []
            for x in X:
                s = np.array([0, 0, x, 0, 0])
                t = np.array(y, ndmin=1)
                st = np.concatenate((s, t))
                l.append(self.models[a].predict(st)[0])
            mesh.append(l)
        return mesh

    def plot_pred(self, data):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        # Generate points
        a = 0
        x = []
        t = []
        q = []
        for d in data:
            if a == d[2]:
                x.append(d[0][2]) # select theta
                t.append(d[1]) # time
                q.append(d[3]) # qval
        # Generate slope
        xg = np.arange(-0.1, 0.1, 0.01)
        tg = np.arange(0, 0.2, 0.01)
        qpred = np.array(self.plpred(xg, tg, a))
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, t, q, c='r', marker='o')
        xg, tg = np.meshgrid(xg, tg)
        surf = ax.plot_surface(xg, tg, qpred, cmap=cm.Blues, alpha=0.5, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('Q')
        plt.show()

    def print_info(self):
        for i in range(len(self.models)):
            print('model {} trained with {} data pts'.format(i,self.models[i].n_data))

