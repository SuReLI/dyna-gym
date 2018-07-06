"""
A Random Agent given as an example
"""

class MyRandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        '''
        Reset Agent's attributes.
        Nothing to reset.
        '''

    def act(self, observation=None, reward=None, done=None):
        return self.action_space.sample()
