import numpy as np
from ..SimEnv.Env1110 import Env

class SingleSessionEnv:
    """Wraps Env to expose a simple discrete action interface for one session."""
    def __init__(self, path_pre, file_name, topo_name, tm_circle=10, len_circle=100, seed=66):
        self.path_pre = path_pre
        self.file_name = file_name
        self.topo_name = topo_name
        self.tm_circle = tm_circle
        self.len_circle = len_circle
        self.seed = seed
        self.env = None
        self.pathnum = None
        self.n_actions = None
        self.state_dim = None

    def _init_env(self):
        self.env = Env(self.path_pre, self.file_name, self.topo_name,
                        self.tm_circle * self.len_circle, self.seed,
                        0, self.tm_circle, self.len_circle, 0)
        _, sess_num, _, self.pathnum, _, _ = self.env.getInfo()
        assert sess_num > 0, "environment has no sessions"
        self.n_actions = self.pathnum[0]

    def reset(self):
        self._init_env()
        # start with equal split
        default_act = []
        for n in self.pathnum:
            default_act.extend([1.0 / n] * n)
        max_util, _, net_util, tm = self.env.update_sol10(default_act)
        state = self._build_state(net_util, tm)
        return state

    def step(self, action):
        # build full action vector: choose path for session0, others equal split
        act_vec = []
        for i, n in enumerate(self.pathnum):
            if i == 0:
                for p in range(n):
                    act_vec.append(1.0 if p == action else 0.0)
            else:
                act_vec.extend([1.0 / n] * n)
        max_util, _, net_util, tm = self.env.update_sol10(act_vec)
        next_state = self._build_state(net_util, tm)
        reward = -max_util
        done = False
        return next_state, reward, done, {}

    def _build_state(self, net_util, tm):
        flat_util = [u for sub in net_util for u in sub]
        state = np.array(flat_util + tm, dtype=np.uint8)
        if self.state_dim is None:
            self.state_dim = len(state)
        return state
