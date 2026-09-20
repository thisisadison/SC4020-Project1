import numpy as np

from sklearn.datasets._samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

VERBOSE = False  # set True to see the per-step prints



# get fake vector db and query for search
# xb is for vector db
# xq is for query
def generate_random_data(nb=150000, nq=100, d=64):
    """
    Generate numpy array for vector db and query
    
    :param nb: no of vectors in vector db
    :param nq: no of queries
    :param d: dimension of vectors 
    :return: xb numpy array for vector db, xq numpy array for query
    """
    if VERBOSE:
        print("\n=== Generate Random Data ===")

    # float32 for faiss
    np.random.seed(1234)  # make reproducible

    xb_raw = np.random.randn(nb, d)
    xq_raw = np.random.randn(nq, d)

    xb = normalize(xb_raw, norm='l2').astype('float32')
    xq = normalize(xq_raw, norm='l2').astype('float32')

    return xb, xq



def generate_clustered_data(nb=128000, nq=100, centers=6, n_features=64, random_state=42):
    """
    Create clustered data for vector search benchmarking.
    
    :param nb: Number of database/index vectors
    :param nq: Number of query vectors
    :param centers: Number of cluster centers
    :param n_features: Number of features (dimensions) per vector
    :param random_state: Seed for reproducibility
    :return: xb (database array), xq (query array)
    """
    if VERBOSE:
        print("\n=== Generate Clustered Data ===")

    total_samples = nb + nq
    X_raw, y = make_blobs(n_samples=total_samples, centers=centers, n_features=n_features, random_state=random_state)

    np.random.seed(random_state)
    transformation = np.random.randn(n_features, n_features)
    X_correlated = np.dot(X_raw, transformation) # Stretches the perfect spheres into ellipses

    xb, xq = train_test_split(X_correlated, train_size=nb, test_size=nq, shuffle=True, random_state=random_state)

    return xb.astype("float32"), xq.astype("float32")