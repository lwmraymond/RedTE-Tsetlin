import numpy as np
from tmu.models.regression.vanilla_regressor import TMRegressor
from .simple_replay import SimpleReplayBuffer

class TsetlinQNetwork:
    """Q network based on Tsetlin Machines."""

    def __init__(self, input_dim, n_actions, clauses=100, T=15, s=3.9):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.models = [TMRegressor(number_of_clauses=clauses, T=T, s=s) for _ in range(n_actions)]
        self.trained = [False] * n_actions

    def predict(self, X):
        """Predict Q-values for all actions."""
        X = np.asarray(X, dtype=np.uint8)
        q_vals = []
        for a, model in enumerate(self.models):
            if self.trained[a]:
                q_vals.append(model.predict(X).ravel())
            else:
                q_vals.append(np.zeros(len(X)))
        return np.stack(q_vals, axis=1)

    def update(self, X, actions, targets):
        X = np.asarray(X, dtype=np.uint8)
        actions = np.asarray(actions)
        targets = np.asarray(targets)
        for a in range(self.n_actions):
            idx = np.where(actions == a)[0]
            if idx.size:
                self.models[a].fit(X[idx], targets[idx], shuffle=False)
                self.trained[a] = True

    def copy_from(self, other):
        for a in range(self.n_actions):
            self.models[a] = other.models[a]

class TsetlinDQNAgent:
    def __init__(self, state_dim, n_actions, buffer_size=10000, batch_size=32,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 clauses=100, T=15, s=3.9):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = SimpleReplayBuffer(buffer_size)
        self.q_network = TsetlinQNetwork(state_dim, n_actions, clauses, T, s)
        self.target_network = TsetlinQNetwork(state_dim, n_actions, clauses, T, s)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.q_network.predict([state])[0]
        return int(np.argmax(q_vals))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample_batch(self.batch_size)
        next_q = self.target_network.predict(next_states)
        targets = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)
        self.q_network.update(states, actions, targets)

    def update_target(self):
        self.target_network.copy_from(self.q_network)
