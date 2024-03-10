import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from lib import DQN, ReplayMemory, select_action, optimize_model, plot_durations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num-episodes', type=int, default=500,help="set number of training episodes")
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="set training batch size (i.e., the number of experiences sampled from the replay memory)")
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help="set the discount factor")
    parser.add_argument('--eps-start', type=float, default=0.9, help="set the initial value of epsilon")
    parser.add_argument('--eps-end', type=float, default=0.05, help="set the final value of epsilon")
    parser.add_argument('--eps-decay', type=float, default=1000, help="set the rate of exponential decay of epsilon (higher meaning a slower decay)")
    parser.add_argument('--tau', type=float, default=0.005, help="set the update rate of the target network")
    parser.add_argument('--lr', type=float, default=1e-4, help="set the learning rate")

    # Parse given arguments
    args = parser.parse_args()

    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make the CartPole environment
    env = gym.make("CartPole-v1")

    # Get the number of actions from gym action space
    n_actions = env.action_space.n

    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    # Build the networks
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Build optimizer (AdamW)
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)

    # Build Replay Memory
    transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    memory = ReplayMemory(10000, transition=transition)

    # Training loop
    steps_done = 0
    episode_durations = []
    for i_episode in tqdm(range(args.num_episodes), desc="Training: "):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state, env, policy_net, steps_done, args.eps_start, args.eps_end, args.eps_decay, device)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            steps_done += 1

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, transition, policy_net, target_net, optimizer, args.gamma, args.batch_size, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
