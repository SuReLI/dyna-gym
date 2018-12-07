"""
Generic methods for tabular NSMDPs
"""

class TabularNSMDP(object):
    #def __init__(self):

    def equality_operator(self, s1, s2):
        """
        Return True if the input states have the same indexes.
        """
        return (s1.index == s2.index)
