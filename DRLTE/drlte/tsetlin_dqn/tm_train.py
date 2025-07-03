import argparse
import os
import sys

if __package__ is None or __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, ROOT)
    from DRLTE.drlte.tsetlin_dqn.tsetlin_dqn import TsetlinDQNAgent
    from DRLTE.drlte.tsetlin_dqn.tm_env import SingleSessionEnv
else:
    from .tsetlin_dqn import TsetlinDQNAgent
    from .tm_env import SingleSessionEnv


def parse_args():
    p = argparse.ArgumentParser(description="Train Tsetlin DQN using DRLTE env")
    p.add_argument("--file_name", required=True, help="input file name, e.g. Abi_train4000")
    p.add_argument("--topo_name", default=None, help="topology name, defaults to prefix of file_name")
    p.add_argument("--path_pre", default="../inputs/", help="input files directory")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--tm_circle", type=int, default=10)
    p.add_argument("--len_circle", type=int, default=100)
    p.add_argument("--seed", type=int, default=66)
    return p.parse_args()


def main():
    args = parse_args()
    topo = args.topo_name or args.file_name.split("_")[0]
    env = SingleSessionEnv(args.path_pre, args.file_name, topo,
                           args.tm_circle, args.len_circle, args.seed)
    state = env.reset()
    agent = TsetlinDQNAgent(state_dim=len(state), n_actions=env.n_actions)

    for ep in range(args.episodes):
        state = env.reset()
        ep_reward = 0.0
        for _ in range(args.steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            ep_reward += reward
        agent.update_target()
        print(f"Episode {ep} reward={ep_reward:.3f} epsilon={agent.epsilon:.2f}")


if __name__ == "__main__":
    main()
