import numpy as np
import random

word_to_cluster = None
cluster_to_words = None
clue_cache = {}

def build_cluster_maps(clusters):
    global word_to_cluster, cluster_to_words
    word_to_cluster = {}
    cluster_to_words = {}

    for _, row in clusters.iterrows():
        cid = row['cluster_id']
        words = row['words']
        cluster_to_words[cid] = words
        for w in words:
            word_to_cluster[w] = cid


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise Exception(f"DIM MISMATCH: {len(a)} vs {len(b)}")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def find_cluster(target_word, clusters):
    for idx, row in clusters.iterrows():
        if target_word in row['words']:
            return row['cluster_id']
    return None


def rank_words(target_word, clusters, cluster_id, embeddings_df):
    # use precomputed map
    list_of_words = cluster_to_words[cluster_id]

    # drop the target
    words = [word for word in list_of_words if word != target_word]
    if not words:
        return [], []

    embedding_vector = embeddings_df[target_word]

    # filter once
    valid_words = [w for w in words if w in embeddings_df]
    other_embeddings = np.array([embeddings_df[w] for w in valid_words])
    words = valid_words

    if len(other_embeddings) == 0:
        return [], []

    # vectorize cosine for speed
    embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
    other_embeddings = other_embeddings / np.linalg.norm(other_embeddings, axis=1, keepdims=True)

    similarities = other_embeddings @ embedding_vector

    # rank words by similarity
    ranked_pairs = sorted(zip(similarities, words), reverse=True)
    ranked_words = [word for _, word in ranked_pairs]
    ranked_similarities = [sim for sim, _ in ranked_pairs]

    return ranked_words, ranked_similarities


def choose_clue(ranked_words, similarities, drop_pct):
    num_to_drop = max(int(len(ranked_words) * drop_pct), 0)

    remaining_words = ranked_words[num_to_drop:]
    remaining_similarities = similarities[num_to_drop:]

    if not remaining_words:
        return None

    probabilities = softmax(np.array(remaining_similarities))
    clue = random.choices(remaining_words, weights=probabilities, k=1)[0]

    return clue


def get_clue(target_word, clusters, embeddings):
    # cache to avoid recomputation
    if target_word in clue_cache:
        return clue_cache[target_word]

    cluster_id = word_to_cluster.get(target_word)
    if cluster_id is None:
        return None

    ranked_words, similarities = rank_words(target_word, clusters, cluster_id, embeddings)
    clue = choose_clue(ranked_words, similarities, drop_pct=0)

    clue_cache[target_word] = clue
    return clue


def get_n_clues(target_word, clusters, n, embeddings):
    clues = []
    for i in range(n):
        clues.append(get_clue(target_word, clusters, embeddings))
    print("Obtained clues: ", clues)
    return set(clues)