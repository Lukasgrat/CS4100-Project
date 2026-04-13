import numpy as np
import random
from sentence_transformers import SentenceTransformer

# 1. find cluster the word belongs to
# 2. rank word similairities
# 3. drop the first 2 (or maybe make it a percentage) words that are too similar
# 4. softmax over probs
# 5. sample from dist of words -> the clue

# cosine similarity
def cosine_similarity(a, b):
    if len(a) != len(b):
        raise Exception(f"DIM MISMATCH: {len(a)} vs {len(b)}")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# getting the cluster id for the given word
def find_cluster(target_word, clusters):
    for idx, row in clusters.iterrows():
        if target_word in row['words']:
            return row['cluster_id']
    return None

# rank words by similarity for a given cluster
# returns a list of words ranked by similarity (excluding the target) and the similarities for those words
def rank_words(target_word, clusters, cluster_id, embeddings_df):
    row = clusters[clusters['cluster_id'] == cluster_id].iloc[0]
    list_of_words = row['words']

    # drop the target
    words = [word for word in list_of_words if word != target_word]
    if not words:
        return [], []

    embedding_vector = embeddings_df[target_word]

    other_embeddings = np.array([embeddings_df[w] for w in words if w in embeddings_df])
    words = [w for w in words if w in embeddings_df]

    # compute cosine similarity row by row
    similarities = [cosine_similarity(embedding_vector, emb) for emb in other_embeddings]

    # rank words by similarity
    ranked_words = [word for _, word in sorted(zip(similarities, words), reverse=True)]
    ranked_similarities = sorted(similarities, reverse=True)

    return ranked_words, ranked_similarities

# choose the clues
# given a list of potential words to choose, drop too similar words and softmax over the rest
# and sample from that to choose the clue
def choose_clue(ranked_words, similarities, drop_pct):
    # drop the top drop_pct% of words
    num_to_drop = max(int(len(ranked_words) * drop_pct), 0)
    remaining_words = ranked_words[num_to_drop:]
    remaining_similarities = similarities[num_to_drop:]

    if not remaining_words:
        return None

    # softmax over similarities to get probabilities
    probabilities = softmax(np.array(remaining_similarities))

    # sample from the distribution to choose a clue (k = 1 since only 1 word)
    clue = random.choices(remaining_words, weights=probabilities, k=1)[0]

    return clue

# main function to get the clue for a given word
def get_clue(target_word, clusters, embeddings):
    cluster_id = find_cluster(target_word, clusters)
    if cluster_id is None:
        # TODO
        print("Invalid cluster_id")
        # might want to make this return a default clue later
        return None

    ranked_words, similarities = rank_words(target_word, clusters, cluster_id, embeddings)
    # for now, drop 0%
    clue = choose_clue(ranked_words, similarities, drop_pct=0)
    return clue

# when there are multiple clue givers
def get_n_clues(target_word, clusters, n, embeddings):
    clues = []
    for i in range(n):
        clues.append(get_clue(target_word, clusters, embeddings))
    #print("Obtained clues: ", clues)
    # return no duplicates
    return set(clues)