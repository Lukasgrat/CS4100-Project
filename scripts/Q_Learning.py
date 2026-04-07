import random
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from environment import GameModelEnv, GameModel


BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

env = GameModelEnv() # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hasObs(obs):
	'''
  Idea: Guesses shouldn't have an order, and we need consistency with the list when converting to the tuple, so sorting for the hash solves both
	
	'''
	return hash(tuple(sorted(obs)))
def hashAction(action, env : GameModelEnv):
	for x in range(0, len(env.model.words)):
		if(env.model.words[x] == action):
			return x
	return -1
def Q_learning(clues : set[str], num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
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
	for iteration in range(0, num_episodes):
		
		#Reset for the next run
		in_completion_state = False
		current_Observation = env.reset(clues)
		total_reward = 0
		hashed_state = hash(list(current_Observation).sort())
		if hashed_state not in Q_table:
			Q_table[hashed_state] = np.zeros(len(env.action_space))
			updateNumber_Table[hashed_state] = np.zeros(len(env.action_space))
		if np.random.rand() > cur_epsilon:
			action = np.argmax(Q_table[hashed_state])
		else:
			action = random.choice(env.action_space)
		new_reward = env.step(action)
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

num_episodes = 5000000
decay_rate = 0.9999995


def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)
def conduct_evaluations(clues : set[str]):
	rewards = []
	# adding metrics to print
	total_steps = 0
	new_states = set()
	total_actions = 0
	actions = 0
	random_actions = 0
	start_time = time.time()




	filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle'
	input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
	Q_table = np.load(filename, allow_pickle=True)

	EVAL_EPISODE_COUNT = 10000
	for ep_number in tqdm(range(EVAL_EPISODE_COUNT)):
		obs, reward, done, info = env.reset(clues)
		total_reward = 0
		steps = 0
		
		while not done:
			state = hash(obs)
			try:
				action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))  # Select action using softmax over Q-values
				actions += 1
			except KeyError:
				action = env.action_space.sample()  # Fallback to random action if state not in Q-table
				random_actions += 1
				new_states.add(state)
			
			obs, reward, done, info = env.step(action)
			steps += 1
			total_actions += 1

			total_reward += reward

		rewards.append(total_reward)
		total_steps += steps
		tqdm.update(ep_number)

	avg_reward = sum(rewards)/len(rewards)
	return avg_reward
def Q_learning_main(train_flag: bool, clues : set[str]):
	if not train_flag:
		return conduct_evaluations(clues)
	if train_flag:
		Q_table = Q_learning(clues, num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning
		# Save the Q-table dict to a file
		with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
			pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)