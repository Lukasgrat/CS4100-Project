import pandas as pd
from scripts.Q_Learning import Q_learning_main
from scripts.choose_clues import get_n_clues
import numpy as np

# pseudocode plan for now
def main():
    # load clusters.pkl
    print("Reading clusters")
    clusters = pd.read_pickle("data/cluster.pkl")
    # load word embeddings
    embeddings = pd.read_pickle("data/embeddings.pkl")

    # randomly choose a word from words.txt
    with open("data/words.txt", "r") as f:
        words = f.read().splitlines()
    target_word = np.random.choice(words)

    # get n amount of clues (set) from the clue givers
    # assume 2 clue givers for now
    print("Getting clues")
    clues = get_n_clues(target_word, clusters, 2, embeddings)
    print("Doing Q-learning")
    # feed these clues into q learning clue guesser
    # keep track of rewards and metrics
    print("Using clues: ", clues)
    Q_learning_main(True, clues)
    

if __name__  == "__main__":
    main()