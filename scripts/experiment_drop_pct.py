"""
Experiment: how does drop_pct affect DQN performance?

For each drop_pct value, trains a fresh DQN and evaluates it,
then plots correct guesses per 1000 eval episodes vs drop_pct.

Run from project root:
    python3 -m scripts.experiment_drop_pct
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from tqdm import tqdm

from environment import GameModelEnv
from scripts.choose_clues import get_n_clues
from scripts.DQN_Learning import (
    DQN, ReplayBuffer, encode_state, word_to_idx,
    NUM_WORDS, STATE_DIM, EMBEDDING_DIM, NUM_CLUES
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DROP_PCT_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TRAIN_EPISODES  = 10_000   # per drop_pct value
EVAL_EPISODES   = 500
GAMMA           = 0.9
LR              = 1e-3
BATCH_SIZE      = 64
TARGET_UPDATE   = 200
EPSILON_START   = 1.0
DECAY_RATE      = 0.9997   # reaches ~0.1 by end of 20k episodes


def load_data():
    clusters = pd.read_pickle("data/cluster.pkl")
    embeddings_df = pd.read_pickle("data/embeddings.pkl")
    embeddings = {row["word"]: row.drop("word").values for _, row in embeddings_df.iterrows()}
    return clusters, embeddings


def train(clusters, embeddings, drop_pct: float) -> nn.Module:
    env = GameModelEnv()
    policy_net = DQN(STATE_DIM, NUM_WORDS)
    target_net = DQN(STATE_DIM, NUM_WORDS)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(capacity=10_000)
    cur_epsilon = EPSILON_START

    for episode in range(TRAIN_EPISODES):
        clues = set()
        while not clues:
            env.reset()
            clues = get_n_clues(env.model.answer, clusters, 2, embeddings, drop_pct=drop_pct)

        env.start_guessing(clues)
        state = encode_state(clues, embeddings)

        avail = [word_to_idx(a) for a in env.action_space]
        avail = [i for i in avail if i >= 0]

        if random.random() > cur_epsilon and avail:
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
            action_idx = avail[int(torch.argmax(q_vals[avail]))]
        else:
            action_idx = random.choice(avail) if avail else 0

        action = env.model.words[action_idx]
        reward = env.step(action, clusters=clusters)

        next_state = np.zeros(STATE_DIM, dtype=np.float32)
        buffer.push(state, action_idx, reward, next_state)
        cur_epsilon *= DECAY_RATE

        if len(buffer) >= BATCH_SIZE:
            s, a, r, ns = buffer.sample(BATCH_SIZE)
            with torch.no_grad():
                target_q = r + GAMMA * target_net(ns).max(dim=1).values
            current_q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (episode + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net


def evaluate(policy_net: nn.Module, clusters, embeddings, drop_pct: float) -> int:
    env = GameModelEnv()
    policy_net.eval()
    correct = 0

    for _ in range(EVAL_EPISODES):
        clues = set()
        while not clues:
            env.reset()
            clues = get_n_clues(env.model.answer, clusters, 2, embeddings, drop_pct=drop_pct)

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

        if env.step(action) == 50:
            correct += 1

    return correct


def run():
    print("Loading data...")
    clusters, embeddings = load_data()

    results = {}
    for drop_pct in DROP_PCT_VALUES:
        print(f"\n--- drop_pct={drop_pct} ---")
        print(f"  Training for {TRAIN_EPISODES} episodes...")
        policy_net = train(clusters, embeddings, drop_pct)
        correct = evaluate(policy_net, clusters, embeddings, drop_pct)
        results[drop_pct] = correct
        print(f"  Correct: {correct}/{EVAL_EPISODES}")

    # Plot
    xs = list(results.keys())
    ys = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker='o', linewidth=2, color='steelblue')
    for x, y in zip(xs, ys):
        plt.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8), ha='center')
    plt.axhline(y=EVAL_EPISODES / NUM_WORDS, color='gray', linestyle='--', label='Random baseline')
    plt.title(f"DQN Correct Guesses vs drop_pct\n({TRAIN_EPISODES} train eps, {EVAL_EPISODES} eval eps)")
    plt.xlabel("drop_pct")
    plt.ylabel(f"Correct guesses / {EVAL_EPISODES}")
    plt.xticks(xs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("drop_pct_experiment.png", dpi=150)
    plt.show()
    print("\nPlot saved to drop_pct_experiment.png")
    print("\nResults:", results)


if __name__ == "__main__":
    run()
