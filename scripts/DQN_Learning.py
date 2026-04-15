import random
import sys
import time
import pickle
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from environment import GameModelEnv
from scripts.choose_clues import get_n_clues

BOLD = '\033[1m'
RESET = '\033[0m'

env = GameModelEnv()
NUM_WORDS = len(env.model.words)
EMBEDDING_DIM = 384       # all-MiniLM-L6-v2 output size
NUM_CLUES = 2             # clue givers per round
STATE_DIM = EMBEDDING_DIM * (NUM_CLUES + 1)  # concat(sorted clues) + mean

num_episodes = 100000
decay_rate = 0.9999


# ---------------------------------------------------------------------------
# Neural network: maps avg(clue embeddings) -> Q-values over all words
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state):
        self.buffer.append((state, action_idx, reward, next_state))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# State encoding: average of clue embedding vectors
# ---------------------------------------------------------------------------
def encode_state(clues, embeddings: dict) -> np.ndarray:
    """
    State = sorted clue embeddings (padded to NUM_CLUES) + their mean.
    Sorting ensures order-invariance; mean adds a global summary signal.
    Fixed size regardless of how many unique clues survived deduplication.
    """
    vecs = [embeddings[str(c)] for c in clues if str(c) in embeddings]
    if not vecs:
        return np.zeros(STATE_DIM, dtype=np.float32)
    # Sort by first dimension for order-invariance, pad to NUM_CLUES
    vecs_sorted = sorted(vecs, key=lambda v: v[0])
    while len(vecs_sorted) < NUM_CLUES:
        vecs_sorted.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
    vecs_sorted = vecs_sorted[:NUM_CLUES]
    mean_vec = np.mean(vecs, axis=0)
    return np.concatenate(vecs_sorted + [mean_vec]).astype(np.float32)


def word_to_idx(word: str) -> int:
    try:
        return env.model.words.index(str(word))
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def DQN_learning(
    clusters,
    embeddings,
    num_episodes: int = 10_000,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    decay_rate: float = 0.999,
    lr: float = 1e-3,
    batch_size: int = 64,
    target_update_freq: int = 200,
):
    policy_net = DQN(STATE_DIM, NUM_WORDS)
    target_net = DQN(STATE_DIM, NUM_WORDS)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(capacity=10_000)

    rewards = []
    cur_epsilon = epsilon
    correct_guesses = 0

    for episode in tqdm(range(num_episodes)):
        clues = set()
        while not clues:
            answer = env.reset()
            clues = get_n_clues(answer, clusters, 2, embeddings)

        env.start_guessing(clues)
        state = encode_state(clues, embeddings)

        avail = [word_to_idx(a) for a in env.action_space]
        avail = [i for i in avail if i >= 0]

        # Epsilon-greedy action selection over available actions only
        if random.random() > cur_epsilon and avail:
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
            action_idx = avail[int(torch.argmax(q_vals[avail]))]
        else:
            action_idx = random.choice(avail) if avail else 0

        action = env.model.words[action_idx]
        reward = env.step(action, clusters=clusters)
        if reward == 50:
            correct_guesses += 1

        # Next state: single-step episode, so terminal — zero vector
        next_state = np.zeros(STATE_DIM, dtype=np.float32)

        buffer.push(state, action_idx, reward, next_state)
        rewards.append(reward)
        cur_epsilon *= decay_rate

        # Train on a mini-batch
        if len(buffer) >= batch_size:
            s, a, r, ns = buffer.sample(batch_size)

            with torch.no_grad():
                target_q = r + gamma * target_net(ns).max(dim=1).values

            current_q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = loss_fn(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f"Training complete. Correct guesses: {correct_guesses}/{num_episodes} ({100*correct_guesses/num_episodes:.1f}%)")
    return policy_net


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def conduct_evaluations_dqn(clusters, embeddings):
    filename = f'DQN_model_{num_episodes}_{decay_rate}.pt'
    input(f"\n{BOLD}Currently loading DQN model from {filename}{RESET}.\n\nPress Enter to confirm, or Ctrl+C to cancel.\n")
    policy_net = DQN(STATE_DIM, NUM_WORDS)
    policy_net.load_state_dict(torch.load(filename, weights_only=True))
    policy_net.eval()

    EVAL_EPISODE_COUNT = 1000
    rewards = []
    correct = 0
    random_actions = 0

    for _ in tqdm(range(EVAL_EPISODE_COUNT)):
        clues = set()
        while not clues:
            answer = env.reset()
            clues = get_n_clues(answer, clusters, 2, embeddings)

        env.start_guessing(clues)
        state = encode_state(clues, embeddings)

        avail = [word_to_idx(a) for a in env.action_space]
        avail = [i for i in avail if i >= 0]

        if avail:
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
            action_idx = avail[int(torch.argmax(q_vals[avail]))]
            action = env.model.words[action_idx]
        else:
            action = random.choice(env.action_space)
            random_actions += 1

        print(f"Clues: {set(str(c) for c in clues)} | Guessed: {action} | Answer: {env.model.answer}")
        reward = env.step(action)
        if reward > 0:
            correct += 1
            print("Guess succeeded!")
        else:
            print("Guess failed.")
        rewards.append(reward)

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nAvg reward: {avg_reward:.2f} | Correct: {correct}/{EVAL_EPISODE_COUNT} | Random fallbacks: {random_actions}")

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    window = max(1, len(rewards) // 10)
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(avg, linewidth=2, label=f"Moving Avg. (w={window})")
    plt.title("DQN Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("dqn_rewards.png", dpi=300)
    plt.show()
    return avg_reward


def DQN_learning_main(train_flag: bool, clusters, embeddings):
    if train_flag:
        print("Beginning DQN training")
        policy_net = DQN_learning(
            clusters, embeddings,
            num_episodes=num_episodes,
            gamma=0.9,
            epsilon=1.0,
            decay_rate=decay_rate,
        )
        torch.save(policy_net.state_dict(), f'DQN_model_{num_episodes}_{decay_rate}.pt')
        print(f"Model saved to DQN_model_{num_episodes}_{decay_rate}.pt")
    else:
        return conduct_evaluations_dqn(clusters, embeddings)
