from lwpr import LWPR

class LWPRForIQUCT(object):
    '''
    LWPR model for approximated IQUCT model
    '''
    def __init__(self):
        print('foo')
        #TODO

    def update(self, data):
        '''
        Update the model wrt to the input data.
        Data points should have the form (s, t, a, Q) where Q is the predicted value
        '''
        ''' #TRM
        print('registered {} data pts'.format(len(data)))#TRM
        print('1st data points:')#TRM
        for i in range(10):#TRM
            print(data[i])
        '''
        #TODO

    def prediction_at(self, s, t, a):
        '''
        Compute a prediction at the given input
        '''
        #TODO
        return 0.0
