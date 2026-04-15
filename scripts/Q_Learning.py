import random
import sys
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from environment import GameModelEnv, GameModel
from scripts.choose_clues import *

BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

env = GameModelEnv() # Gym environment already initialized within vis_gym.py

NUM_WORDS = len(env.model.words)  # Fixed action space size (633)

def hashObs(obs, clusters):
	'''
	State = sorted tuple of cluster IDs for each clue.
	Same clusters in any order → same state.
	Converts to plain str to avoid np.str_ vs str hash mismatches.
	'''
	cluster_ids = tuple(sorted(find_cluster(str(v), clusters) for v in obs))
	return hash(cluster_ids)

def hashAction(action):
	'''Maps a word to its fixed index in env.model.words (0 to NUM_WORDS-1).'''
	try:
		return env.model.words.index(str(action))
	except ValueError:
		return -1
def Q_learning(clusters, embeddings, num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	Q_table = {}
	updateNumber_Table = {}
	rewards = []
	cur_epsilon = epsilon
	correct_guesses = 0

	for _ in tqdm(range(num_episodes)):
		clues = set()
		while not clues:
			answer = env.reset()
			clues = get_n_clues(answer, clusters, 2, embeddings)
		env.start_guessing(clues)

		hashed_state = hashObs(clues, clusters)
		if hashed_state not in Q_table:
			Q_table[hashed_state] = np.zeros(NUM_WORDS)
			updateNumber_Table[hashed_state] = np.zeros(NUM_WORDS)

		# Available action indices (words not used as clues)
		avail = [hashAction(a) for a in env.action_space]
		avail = [i for i in avail if i >= 0]

		if np.random.rand() > cur_epsilon and avail:
			# Greedy: best Q-value among available actions only
			action_idx = avail[int(np.argmax(Q_table[hashed_state][avail]))]
			action = env.model.words[action_idx]
		else:
			action = random.choice(env.action_space)
			action_idx = hashAction(action)

		new_reward = env.step(action)
		if new_reward > 0:
			correct_guesses += 1

		if action_idx >= 0:
			η = 1 / (1 + updateNumber_Table[hashed_state][action_idx])
			v = np.max(Q_table[hashed_state][avail]) if avail else 0
			Q_table[hashed_state][action_idx] = (
				(1 - η) * Q_table[hashed_state][action_idx]
				+ η * (new_reward + gamma * v)
			)
			updateNumber_Table[hashed_state][action_idx] += 1

		cur_epsilon *= decay_rate
		rewards.append(new_reward)

	print(f"Training complete. Correct guesses: {correct_guesses}/{num_episodes} ({100*correct_guesses/num_episodes:.1f}%)")
	return Q_table

'''
Specify number of episodes and decay rate for training and evaluation.
'''

num_episodes = 100000
decay_rate = 0.9999


def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)
def conduct_evaluations(clusters, embeddings):
	rewards = []
	new_states = set()
	actions = 0
	random_actions = 0
	correct = 0

	filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle'
	input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
	Q_table = np.load(filename, allow_pickle=True)

	EVAL_EPISODE_COUNT = 1000
	for _ in tqdm(range(EVAL_EPISODE_COUNT)):
		clues = set()
		while not clues:
			answer = env.reset()
			clues = get_n_clues(answer, clusters, 2, embeddings)
		env.start_guessing(clues)
		hashed_state = hashObs(clues, clusters)
		avail = [hashAction(a) for a in env.action_space]
		avail = [i for i in avail if i >= 0]
		try:
			q_vals = softmax(Q_table[hashed_state][avail])
			action_idx = avail[np.random.choice(len(avail), p=q_vals)]
			action = env.model.words[action_idx]
			actions += 1
		except KeyError:
			action = np.random.choice(env.action_space)
			random_actions += 1
			new_states.add(hashed_state)
		print(f"Clues: {set(str(c) for c in clues)} | Guessed: {action} | Answer: {env.model.answer}")
		reward = env.step(action)
		if reward == 50:
			correct += 1
			print("Guess succeeded!")
		else:
			print("Guess failed.")
		rewards.append(reward)

	print(f"\nAvg reward: {sum(rewards)/len(rewards):.2f} | Correct: {correct}/{EVAL_EPISODE_COUNT} | Random fallbacks: {random_actions}")

	avg_reward = sum(rewards)/len(rewards)
	plt.figure(figsize=(10, 6))
	plt.plot(rewards, alpha=0.3, label="Episode Reward")
	window = max(1, len(rewards) // 10)
	if len(rewards) >= window:
		avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
		plt.plot(avg, linewidth=2, label=f"Moving Avg. (w={window})")
	plt.title("Rewards per Episode")
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.savefig("q_learning_rewards.png", dpi=300)
	plt.show()
	return avg_reward
def Q_learning_main(train_flag: bool, clusters, embeddings):
	if not train_flag:
		return conduct_evaluations(clusters, embeddings)
	if train_flag:
		print("Beginning Q-learning")
		Q_table = Q_learning(clusters, embeddings, num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning
		# Save the Q-table dict to a file
		with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
			pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)