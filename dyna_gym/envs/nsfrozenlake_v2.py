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

class State:
    """
    State class
    """
    def __init__(self, index, time):
        self.index = index
        self.time = time

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

class NSFrozenLakeV2(Env):
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

    Non-Stationarity: when the transition function is stochastic , i.e. slippery ice,
    the probability of the resulting states from any action evolves randomly through
    time. The resulting transition function is L_p-Lipschitz.

    The reward function is non-stationary: random reward received anywhere and evolving randomly.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", map_size=(5,5), is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            if map_name is "random":
                desc = random_map(map_size)
            else:
                desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.nS = nrow * ncol # n states
        self.nA = 4 # n actions
        self.nT = 21 # n timesteps
        self.action_space = spaces.Discrete(self.nA)
        self.is_slippery = is_slippery
        self.tau = 1 # timestep duration
        self.L_p = 1.0
        self.L_r = 0.1
        self.r_goal_max = 1.0
        self.r_goal_min = 0.1
        #self.T = self.generate_transition_matrix()
        #self.R = self.generate_instant_reward_matrix()
        isd = np.array(self.desc == b'S').astype('float64').ravel() # Initial state distribution
        self.isd = isd / isd.sum()
        #self._seed()
        self.np_random = np.random.RandomState()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment.
        IMPORTANT: Does not create a new environment.
        """
        self.state = State(categorical_sample(self.isd, self.np_random), 0) # (index, time)
        self.lastaction = None # for rendering
        self.T = self.generate_transition_matrix()
        self.R = self.generate_instant_reward_matrix()
        return self.state

    def display(self):
        print('Displaying NSFrozenLakeEnv-v2')
        print('map       :')
        print(self.desc)
        print('n states  :', self.nS)
        print('n actions :', self.nA)
        print('timeout   :', self.nT)

    def inc(self, row, col, a):
        """
        Given a position (row, col) and an action a, return the resulting position (row, col).
        """
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
        """
        From the state's position (row, col), retrieve the state index.
        """
        return row * self.ncol + col

    def to_m(self, s):
        """
        From the state index, retrieve the state's position (row, col).
        """
        row = int(s / self.ncol)
        col = s - row * self.ncol
        return row, col

    def distance(self, s1, s2):
        """
        Return the Manhattan distance between the positions of states s1 and s2
        """
        if (type(s1) == State) and (type(s2) == State):
            row1, col1 = self.to_m(s1.index)
            row2, col2 = self.to_m(s2.index)
        else:
            assert (type(s1) == int), 'Error: input state has wrong type: type={}'.format(type(s1))
            assert (type(s2) == int), 'Error: input state has wrong type: type={}'.format(type(s2))
            row1, col1 = self.to_m(s1)
            row2, col2 = self.to_m(s2)
        return abs(row1 - row2) + abs(col1 - col2)

    def equality_operator(self, s1, s2):
        """
        Return True if the input states have the same indexes.
        """
        return (s1.index == s2.index)

    def reachable_states(self, s, a):
        if (type(s) == State):
            row, col = self.to_m(s.index)
        else:
            assert (type(s) == int), 'Error: input state has wrong type: type={}'.format(type(s))
            row, col = self.to_m(s)
        rs = np.zeros(shape=self.nS, dtype=int)
        if self.is_slippery:
            #for b in [(a-1)%4, a, (a+1)%4]:# Put back for 3 reachable states
            for b in range(4):
                newrow, newcol = self.inc(row, col, b)
                rs[self.to_s(newrow, newcol)] = 1
        else:
            newrow, newcol = self.inc(row, col, a)
            rs[self.to_s(newrow, newcol)] = 1
        return rs

    def distances_matrix(self, states):
        """
        Return the distance matrix D corresponding to the states of the input array.
        D[i,j] = distance(si, sj)
        """
        n = len(states)
        D = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i+1, n):
                D[i,j] = self.distance(states[i], states[j])
                D[j,i] = self.distance(states[i], states[j])
        return D

    def generate_instant_reward_matrix(self):
        R = np.zeros(shape=(self.nS, self.nT), dtype=float)
        for i in range(self.nS):
            row, col = self.to_m(i)
            letter = self.desc[row, col]
            if letter == b'G':
                R[i][0] = np.random.uniform(low=-self.r_goal_max, high=self.r_goal_max)
                for j in range(1, self.nT):
                    R[i][j] = R[i][j-1] + np.random.uniform(low=-self.tau * self.L_r, high=self.tau * self.L_r)
                    if R[i][j] > self.r_goal_max:
                        R[i][j] = self.r_goal_max
                    elif R[i][j] < self.r_goal_min:
                        R[i][j] = self.r_goal_min
        return R

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for s in range(self.nS):
            for a in range(self.nA):
                # Generate distribution for t=0
                rs = self.reachable_states(s, a)
                nrs = np.sum(rs)
                w = distribution.random_tabular(size=nrs)
                wcopy = list(w.copy())
                T[s,a,0,:] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)
                row, col = self.to_m(s)
                row_p, col_p = self.inc(row, col, a)
                s_p = self.to_s(row_p, col_p)
                T[s,a,0,s_p] += 1.0 # Increase weight on normally reached state
                T[s,a,0,:] /= sum(T[s,a,0,:])
                states = []
                for k in range(len(rs)):
                    if rs[k] == 1:
                        states.append(State(k,0))
                D = self.distances_matrix(states)
                # Build subsequent distributions st LC constraint is respected
                for t in range(1, self.nT): # t
                    w = distribution.random_constrained(w, D, self.L_p * self.tau)
                    wcopy = list(w.copy())
                    T[s,a,t,:] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)
        return T

    def transition_probability_distribution(self, s, t, a):
        assert s.index < self.nS, 'Error: index bigger than nS: s.index={} nS={}'.format(s.index, nS)
        assert t < self.nT, 'Error: time bigger than nT: t={} nT={}'.format(t, self.nT)
        assert a < self.nA, 'Error: action bigger than nA: a={} nA={}'.format(a, nA)
        return self.T[s.index, a, t]

    def transition_probability(self, s_p, s, t, a):
        assert s_p.index < self.nS, 'Error: position bigger than nS: s_p.index={} nS={}'.format(s_p.index, nS)
        assert s.index < self.nS, 'Error: position bigger than nS: s.index={} nS={}'.format(s.index, nS)
        assert t < self.nT, 'Error: time bigger than nT: t={} nT={}'.format(t, self.nT)
        assert a < self.nA, 'Error: action bigger than nA: a={} nA={}'.format(a, nA)
        return self.T[s.index, a, t, s_p.index]

    def get_time(self):
        return self.state.time

    def dynamic_reachable_states(self, s, a):
        """
        Return a numpy array of the reachable states.
        Dynamic means that time increment is performed.
        """
        rs = self.reachable_states(s, a)
        srs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                srs.append(State(i, s.time + self.tau))
        assert (len(srs) == sum(rs))
        return np.array(srs)

    def static_reachable_states(self, s, a):
        """
        Return a numpy array of the reachable states.
        Static means that no time increment is performed.
        """
        rs = self.reachable_states(s, a)
        srs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                srs.append(State(i, s.time))
        assert (len(srs) == sum(rs))
        return np.array(srs)

    def transition(self, s, a, is_model_dynamic=True):
        """
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        """
        d = self.transition_probability_distribution(s, s.time, a)
        p_p = categorical_sample(d, self.np_random)
        if is_model_dynamic:
            s_p = State(p_p, s.time + self.tau)
        else:
            s_p = State(p_p, s.time)
        r = self.instant_reward(s, s.time, a, s_p)
        done = self.is_terminal(s_p)
        return s_p, r, done

    def instant_reward(self, s, t, a, s_p):
        """
        Return the instant reward for transition s, t, a, s_p
        """
        newrow, newcol = self.to_m(s_p.index)
        newletter = self.desc[newrow, newcol]
        if newletter == b'G':
            return self.R[s_p.index, t]
        elif newletter == b'H':
            return -1.0
        else:
            return 0.0

    def expected_reward(self, s, t, a):
        """
        Return the expected reward function at s, t, a
        """
        R = 0.0
        d = self.transition_probability_distribution(s, t, a)
        for i in range(len(d)):
            s_p = State(i, s.time + self.tau)
            r_i = self.instant_reward(s, t, a, s_p)
            R += r_i * d[i]
        return R

    def is_terminal(self, s):
        """
        Return True if the input state is terminal.
        """
        row, col = self.to_m(s.index)
        letter = self.desc[row, col]
        done = bytes(letter) in b'GH'
        if s.time + self.tau >= self.nT: # Timeout
            done = True
        return done

    def step(self, a):
        s, r, done = self.transition(self.state, a, True)
        self.state = s
        self.lastaction = a
        return (s, r, done, {})

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.state.index // self.ncol, self.state.index % self.ncol
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
