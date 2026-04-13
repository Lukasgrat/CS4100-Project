import pandas as pd
from scripts.Q_Learning import Q_learning_main
from scripts.choose_clues import get_n_clues
import numpy as np

# pseudocode plan for now
def main():
    # load clusters.pkl
    clusters = pd.read_pickle("data/cluster.pkl")

    # load word embeddings
    embeddings = pd.read_pickle("data/embeddings.pkl")
    if isinstance(embeddings, pd.DataFrame):
            embeddings = {
                row['word']: row.drop('word').values
                for _, row in embeddings.iterrows()
            }
    # randomly choose a word from words.txt
    with open("data/words.txt", "r") as f:
        words = f.read().splitlines()
    target_word = np.random.choice(words)

    # get n amount of clues (set) from the clue givers
    # assume 2 clue givers for now

    # feed these clues into q learning clue guesser
    # keep track of rewards and metrics
    avg_reward = Q_learning_main(False, clusters, embeddings)
    print("Ending evaluation with reward: " + str(avg_reward))

if __name__  == "__main__":
    main()