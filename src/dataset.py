import numpy as np

# get fake vector db and query for search
# xb is for vector db
# xq is for query
def generate_fake_data(nb=10000, nq=100, d=64):
    """
    Generate numpy array for vector db and query
    
    :param nb: no of vectors in vector db
    :param nq: no of queries
    :param d: dimension of vectors 
    :return: xb numpy array for vector db, xq numpy array for query
    """

    # float32 for faiss
    np.random.seed(1234)  # make reproducible

    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    return xb, xq