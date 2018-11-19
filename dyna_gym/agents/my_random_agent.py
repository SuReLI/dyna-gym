"""
A Random Agent given as an example
"""

class MyRandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying Random agent:')
        print('Action space       :', self.action_space)

    def reset(self, p):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        assert len(p) == 1, 'Error: expected 1 parameters received {}'.format(len(p))
        assert type(p[0]) == spaces.discrete.Discrete, 'Error: wrong type, expected "gym.spaces.discrete.Discrete", received {}'.format(type(p[0]))
        self.__init__(p[0])

    def act(self, observation=None, reward=None, done=None):
        return self.action_space.sample()
