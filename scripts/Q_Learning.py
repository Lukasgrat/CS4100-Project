import random
import sys
import time
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

#env.render() # Uncomment to print game state info

def hashObs(obs, embeddings, clusters):
	'''
  Idea: Guesses shouldn't have an order, and we need consistency with the list when converting to the tuple, so sorting for the hash solves both
	
	'''
	arr = []
	for val in sorted(obs):
		#print("Putting word: " + val)
		num = find_cluster(val, clusters)
		#print("Number: " + str(num))
		arr.append(num)
	return hash(tuple(arr))
def hashAction(action, env : GameModelEnv):
	for x in range(0, len(env.model.words)):
		if(env.model.words[x] == action):
			return x
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
	corret_guesses = 0
	for iteration in range(0, num_episodes):
		print("Running episode: " +  str(iteration))
		#Reset for the next run
		in_completion_state = False
		answer = env.reset()
		clues = get_n_clues(answer, clusters, 2, embeddings)
		current_Observation = env.start_guessing(clues)

		total_reward = 0
		hashed_state = hashObs(current_Observation, embeddings, clusters)
		if hashed_state not in Q_table:
			Q_table[hashed_state] = np.zeros(len(env.action_space))
			updateNumber_Table[hashed_state] = np.zeros(len(env.action_space))
		if np.random.rand() > cur_epsilon:
			action = env.action_space[np.argmax(Q_table[hashed_state])]
		else:
			action = random.choice(env.action_space)
		new_reward = env.step(action)
		if(new_reward > 0):
			#print("correct guess number: " + str(corret_guesses) + " with word" + action)
			#print("sending correct to " + str(hashed_state))
			corret_guesses += 1
		total_reward += new_reward
		η = 1 / (1 + updateNumber_Table[hashed_state][hashAction(action, env)])
		if hashed_state not in Q_table:
			Q_table[hashed_state] = np.zeros(len(env.action_space))
			updateNumber_Table[hashed_state] = np.zeros(len(env.action_space))
		v = np.max(Q_table[hashed_state])
		Q_table[hashed_state][hashAction(action, env)] = (1 - η) * Q_table[hashed_state][hashAction(action, env)] + η * (new_reward + gamma * v)
		updateNumber_Table[hashed_state][hashAction(action, env)] += 1

		#Increase decay and add in the reward for the current episode
		cur_epsilon *= decay_rate
		rewards.append(total_reward)
	return Q_table

'''
Specify number of episodes and decay rate for training and evaluation.
'''

num_episodes = 10000
decay_rate = 0.999999


def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)
def conduct_evaluations(clusters, embeddings):
	rewards = []
	# adding metrics to print
	total_steps = 0
	new_states = set()
	actions = 0
	random_actions = 0
	start_time = time.time()




	filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle'
	input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
	Q_table = np.load(filename, allow_pickle=True)

	EVAL_EPISODE_COUNT = 1000
	for ep_number in tqdm(range(EVAL_EPISODE_COUNT)):
		answer = env.reset()
		clues = get_n_clues(answer, clusters, 2, embeddings)
		current_Observation = env.start_guessing(clues)
		total_reward = 0
		hashed_state = hashObs(current_Observation, embeddings, clusters)
		try:
			action = np.random.choice(env.action_space, p=softmax(Q_table[hashed_state]))  # Select action using softmax over Q-values
			actions += 1
		except KeyError:
			action = np.random.choice(env.action_space)  # Fallback to random action if state not in Q-table
			random_actions += 1
			new_states.add(hashed_state)
		print("Guessing with action " + str(action) + " actual word is " + env.model.answer)
		reward = env.step(action)
		if(reward == 50):
			print("Guess succeeded!")
		else:
			print("Guess failed...")
		if(ep_number == 100):
			input()
		total_reward += reward

		rewards.append(total_reward)

	avg_reward = sum(rewards)/len(rewards)	# plotting all the rewards
	plt.figure(figsize=(10, 6))
	# AI USAGE: the initial reward graph was super messy and it was very hard to interpret so
	# AI suggsted using the alpha parameter to smooth things over and to take a moving average
	# instead to make the line plot look cleaner
	plt.figure(figsize=(10, 6))
	plt.plot(rewards, alpha=0.3, label="Episode Reward")
	# moving average
	# change window size for every change in training episodes
	window = 100000
	if len(rewards) >= window:
		avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
		plt.plot(avg, linewidth=2, label="Moving Avg.")
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