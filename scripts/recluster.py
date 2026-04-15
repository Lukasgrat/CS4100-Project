"""
Regenerate data/cluster.pkl using k-means instead of HDBSCAN.

K-means assigns every word to a cluster (no noise cluster), so all 633 words
are playable in Just One. Run from the project root:

    python3 -m scripts.recluster [--k 30]
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def build_clusters(k: int = 30) -> pd.DataFrame:
    embeddings_df = pd.read_pickle("data/embeddings.pkl")
    words = embeddings_df["word"].tolist()
    vectors = embeddings_df.drop(columns=["word"]).values.astype(np.float32)

    print(f"Running k-means with k={k} on {len(words)} words...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)

    # Group words by cluster id — same format as the old cluster.pkl
    cluster_map: dict[int, list[str]] = {}
    for word, label in zip(words, labels):
        cluster_map.setdefault(int(label), []).append(word)

    cluster_df = pd.DataFrame({
        "cluster_id": list(cluster_map.keys()),
        "words": list(cluster_map.values()),
    })

    cluster_df.to_pickle("data/cluster.pkl")
    print(f"Saved {len(cluster_df)} clusters to data/cluster.pkl")

    # Print cluster sizes for inspection
    sizes = cluster_df["words"].apply(len).sort_values(ascending=False)
    print(f"Cluster sizes — min: {sizes.min()}  max: {sizes.max()}  mean: {sizes.mean():.1f}")
    for cid, row in cluster_df.iterrows():
        print(f"  Cluster {row['cluster_id']:2d} ({len(row['words'])} words): {row['words'][:6]}")

    return cluster_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=30, help="Number of clusters")
    args = parser.parse_args()
    build_clusters(args.k)
