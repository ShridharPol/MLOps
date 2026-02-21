import pickle
import base64
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def load_data():
    """
    Loads the Iris dataset from sklearn (no CSV needed).
    Returns base64-encoded pickled DataFrame.
    """
    iris = load_iris()
    # Use only the feature columns for clustering
    data = iris.data  # shape: (150, 4)
    print(f"[load_data] Loaded Iris dataset: {data.shape[0]} samples, {data.shape[1]} features")
    serialized = pickle.dumps(data)
    return base64.b64encode(serialized).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data and applies MinMax scaling.
    Returns base64-encoded pickled scaled array.
    """
    data = pickle.loads(base64.b64decode(data_b64))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    print(f"[data_preprocessing] Scaled data shape: {scaled.shape}")
    serialized = pickle.dumps(scaled)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Fits a DBSCAN model on the preprocessed data and saves it.
    DBSCAN automatically determines the number of clusters.
    Returns the labels as a list (JSON-safe).
    """
    data = pickle.loads(base64.b64decode(data_b64))

    # DBSCAN — no need to specify number of clusters
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"[build_save_model] DBSCAN found {n_clusters} clusters, {n_noise} noise points")

    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(dbscan, f)

    return labels.tolist()  # JSON-safe


def evaluate_clusters(data_b64: str, labels: list):
    """
    NEW TASK (not in original lab):
    Evaluates the clustering quality using silhouette score
    and prints per-cluster statistics.
    """
    data = pickle.loads(base64.b64decode(data_b64))
    labels_arr = np.array(labels)

    n_clusters = len(set(labels_arr)) - (1 if -1 in labels_arr else 0)
    n_noise = list(labels_arr).count(-1)

    print(f"[evaluate_clusters] Number of clusters found: {n_clusters}")
    print(f"[evaluate_clusters] Noise points: {n_noise}")

    # Silhouette score (only if more than 1 cluster)
    if n_clusters > 1:
        # Exclude noise points for silhouette calculation
        mask = labels_arr != -1
        score = silhouette_score(data[mask], labels_arr[mask])
        print(f"[evaluate_clusters] Silhouette Score: {score:.4f}")
    else:
        print("[evaluate_clusters] Not enough clusters to compute silhouette score")

    # Per-cluster size
    for cluster_id in sorted(set(labels_arr)):
        count = int((labels_arr == cluster_id).sum())
        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
        print(f"[evaluate_clusters] {label}: {count} points")

    return n_clusters