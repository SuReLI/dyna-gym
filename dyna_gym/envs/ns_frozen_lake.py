import numpy as np
import sys
import dyna_gym.utils.distribution as distribution
from random import randint
from six import StringIO, b
from gym import Env, spaces, utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

def random_map(map_size):
    nR, nC = map_size
    nH = int(0.2 * nR * nC) # Number of holes
    m = []
    for i in range(nR): # Generate ice floe
        m.append(nC * ["F"])
    m[0][0] = "S" # Generate start
    m[-1][-1] = "G" # Generate goal
    while nH > 0: # Generate holes
        i, j = (randint(0, nR-1), randint(0, nC-1))
        if m[i][j] is "F":
            m[i][j] = "H"
            nH -= 1
    for i in range(nR): # Formating
        m[i] = "".join(m[i])
    return m

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class NSFrozenLakeEnv(Env):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="random", map_size=(3,5), is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            if map_name is "random":
                desc = random_map(map_size)
            else:
                desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.nS = nrow * ncol # n states
        self.nA = 4 # n actions
        self.nT = 11 # n timesteps

        self.timestep = 1 # timestep duration
        self.L_p = 0.1 # transition kernel Lipschitz constant
        self.L_r = 10 # reward function Lipschitz constant

        self.action_space = spaces.Discrete(self.nA)
        self.pos_space = np.array(range(self.nS))
        self.observation_space = spaces.Discrete(self.nS)

        self.is_slippery = is_slippery
        self.T = self.generate_transition_matrix()

        isd = np.array(desc == b'S').astype('float64').ravel() # Initial state distribution
        self.isd = isd / isd.sum()

        # TRM P[s][a] == [(probability, nextstate, reward, done), ... for each a]
        #P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        # TRM Fill P
        '''
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH': # is terminal
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]: # for each action 'close' to a
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))
        '''

        #super(NSFrozenLakeEnv, self).__init__(nS, nA, P, isd)#TRM
        #self.P = P #TRM

        self.lastaction=None # for rendering
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = (categorical_sample(self.isd, self.np_random), 0) # (position, time)
        self.lastaction=None # for rendering
        return self.state

    def inc(self, row, col, a):
        if a==0: # left
            col = max(col-1,0)
        elif a==1: # down
            row = min(row+1,self.nrow-1)
        elif a==2: # right
            col = min(col+1,self.ncol-1)
        elif a==3: # up
            row = max(row-1,0)
        return (row, col)

    def to_s(self, row, col):
        return row * self.ncol + col

    def to_m(self, s):
        row = int(s / self.ncol)
        col = s - row * self.ncol
        return row, col

    def reachable_states(self, s, a):
        assert(s >= 0) #TRM
        assert(a >= 0) #TRM
        assert(s < self.nS) #TRM
        assert(a < self.nA) #TRM
        rs = np.zeros(shape=self.nS, dtype=int)
        row, col = self.to_m(s)
        if self.is_slippery:
            for b in [(a-1)%4, a, (a+1)%4]:
                newrow, newcol = self.inc(row, col, b)
                rs[self.to_s(newrow, newcol)] = 1
        else:
            newrow, newcol = self.inc(row, col, a)
            rs[self.to_s(newrow, newcol)] = 1
        return rs

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for i in range(self.nS):
            for j in range(self.nA):
                # Generate distribution for t=0
                rs = self.reachable_states(i, j)
                nrs = np.sum(rs)
                d = distribution.random_tabular(size=nrs)
                dc = list(d.copy())
                T[i,j,0,:] = np.asarray([0 if x == 0 else dc.pop() for x in rs], dtype=float)
                # Build subsequent distributions st LC constraint is respected
                for t in range(1, self.nT): # t
                    d = distribution.random_constrained(d, self.L_p * self.timestep)
                    dc = list(d.copy())
                    T[i,j,t,:] = np.asarray([0 if x == 0 else dc.pop() for x in rs], dtype=float)
        return T

    def transition_probability_distribution(self, p, t, a):
        '''
        Return the distribution of the transition probability conditionned by (p, t, a)
        If a full state (time-enhanced) is provided as argument , only the position is used
        '''
        assert(p < self.nS)
        assert(t < self.nT)
        assert(a < self.nA)
        assert(isinstance(p, np.int64) or isinstance(p, int))
        return self.T[p, a, t]

    def transition_probability(self, p_p, p, t, a):
        '''
        Return the probability of transition to position p_p conditionned by (p, t, a)
        If a full state (time-enhanced) is provided as argument , only the position is used
        '''
        assert(p_p < self.nS)
        assert(p < self.nS)
        assert(t < self.nT)
        assert(a < self.nA)
        assert(isinstance(p, np.int64) or isinstance(p, int))
        assert(isinstance(p_p,np.int64) or isinstance(p_p, int))
        return self.T[p, a, t, p_p]

    def equality_operator(self, s1, s2):
        return (s1 == s2)

    def transition(self, s, a, is_model_dynamic=True):
        '''
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        '''
        #transitions = self.P[s][a]#TRM
        assert(type(s) == tuple) #TRM
        p, t = s # (position, time)
        d = self.transition_probability_distribution(p, t, a)
        p_p = categorical_sample(d, self.np_random)
        newrow, newcol = self.to_m(p_p)
        newletter = self.desc[newrow, newcol]
        done = bytes(newletter) in b'GH'
        r = float(newletter == b'G')
        if is_model_dynamic:
            t += 1
        s_p = (p_p, t)
        return s_p, r, done

    def _step(self, a):
        s, r, d = self.transition(self.state, a, True)
        self.state = s
        self.lastaction = a
        return (s, r, d, {})

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.state[0] // self.ncol, self.state[0] % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
