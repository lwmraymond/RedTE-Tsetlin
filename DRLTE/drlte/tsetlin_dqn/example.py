import os
import sys
import numpy as np

if __package__ is None:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, ROOT)
    from DRLTE.drlte.tsetlin_dqn.tsetlin_dqn import TsetlinDQNAgent
else:
    from .tsetlin_dqn import TsetlinDQNAgent

class DummyRouterEnv:
    """Minimal example environment with a binary state and discrete actions."""
    def __init__(self, n_ports=4, state_dim=8):
        self.n_ports = n_ports
        self.state_dim = state_dim
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.random.randint(2, size=self.state_dim, dtype=np.uint8)

    def step(self, action):
        # Toy transition: next state is random, reward for matching first bit
        self.step_count += 1
        next_state = np.random.randint(2, size=self.state_dim, dtype=np.uint8)
        reward = 1.0 if action == next_state[0] % self.n_ports else 0.0
        done = self.step_count >= 50
        return next_state, reward, done, {}


def main():
    env = DummyRouterEnv()
    agent = TsetlinDQNAgent(state_dim=env.state_dim, n_actions=env.n_ports)

    for episode in range(3):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
        agent.update_target()
        print(f"Episode {episode} finished with epsilon={agent.epsilon:.2f}")


if __name__ == "__main__":
    main()
