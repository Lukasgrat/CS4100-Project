import pandas as pd

# pseudocode plan for now
def main():
    # load clusters.pkl
    clusters = pd.read_pickle("../data/cluster.pkl")

    # load word embeddings
    embeddings = pd.read_pickle("../data/embeddings.pkl")

    # get n amount of clues from the clue givers

    # feed these clues into q learning clue guesser

    # keep track of rewards and metrics


if __name__  == "__main__":
    main()