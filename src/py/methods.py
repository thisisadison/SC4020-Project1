import faiss
import numpy as np



def add_vectors(index, xb: np.ndarray):
    """
    Populate FAISS index with vectors

    :param index: FAISS index
    :param xb: vector db dimensions, np array
    """
    index.add(xb)



def search_index(index, noNeighbors: int, noQueries: int, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors

    :param index: FAISS index
    :param noNeighbors: no of nearest neighbors
    :param noQueries: no of queries
    :param xq: query dimensions, np array
    :return: d distance between query and neighbors, i index of neighbors
    """
    d, i = index.search(xq[:noQueries], noNeighbors)

    return d, i



def build_flat_index(xb: np.ndarray, d: int):
    """
    Build FAISS flat index for vector search using squared l2 distance

    :param xb: index numpy array
    :param d: dimension of vectors
    :return: vector db
    """
    # print("\n=== Build Flat Index ===")
    index = faiss.IndexFlatL2(d);  # build index
    # print("\nTrained:", index.is_trained)

    add_vectors(index, xb)

    return index



def build_pq_index(d: int, m: int, nbits: int, xb: np.ndarray, metric=faiss.METRIC_L2):
    """
    Build compressed vector for Product Quantization: splits each vector into m
    subspaces, learns nbits-worth of centroids per subspace via k-means,
    and stores compressed centroid-ID codes instead of full vectors.

    :param d: dimension of vectors
    :param m: number of subspaces
    :param nbits: bits per subquantizer code
    :param xb: vector db
    :param metric: faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
    :return: trained, empty PQ index
    """
    index = faiss.IndexPQ(d, m, nbits, metric)  # build index
    index.train(xb) # k-means performed, create codebook

    add_vectors(index, xb)

    return index



def build_opq_index(d: int, m: int, nbits: int, xb: np.ndarray):
    """
    Build compressed vector index using Optimized Product Quantization: learns
    a rotation to optimize the vector space, splits each vector into m subspaces,
    learns 2^nbits centroids per subspace via k-means, and stores compressed
    centroid-ID codes instead of full vectors.

    :param d: dimension of vectors
    :param m: number of subspaces
    :param nbits: bits per subquantizer code
    :param xb: vector db
    :return: trained, empty PQ index
    """
    # print("\n=== Build OPQ Index ===")
    index = faiss.index_factory(d, f"OPQ{d},PQ{m}x{nbits}", faiss.METRIC_L2)
    index.train(xb)
    # print("\nTrained:", index.is_trained)

    add_vectors(index, xb)

    return index



def build_lsh_index(d: int, nbits: int, xb: np.ndarray):
    """
    Build LSH index: generates nbits random hyperplanes and hashes each
    vector into an nbits-length binary code based on which side of each
    hyperplane it falls on.

    :param d: dimension of vectors
    :param nbits: number of random hash functions (hyperplanes) / bits per code
    :return: empty LSH index, ready for add_vectors
    """
    # print("\n=== Build LSH Index ===")
    index = faiss.IndexLSH(d, nbits)
    # print("\nTrained:", index.is_trained)

    add_vectors(index, xb)

    return index



def build_flat_index_cosine(xb: np.ndarray, xq: np.ndarray, d: int, k=10):
    """
    Build FAISS flat index for vector search using cosine similarity

    :param xb: index numpy array
    :param xq: query vector numpy array
    :return: cosine_index normalized cosine index, xb_norm normalized vector db, xq_norm normalized vector queries
    """
    # print("\n=== Cosine Similarity Index ===")
    xb_norm = xb.copy()
    xq_norm = xq.copy()
    faiss.normalize_L2(xb_norm)
    faiss.normalize_L2(xq_norm)
    cosine_index = faiss.IndexFlatIP(d)

    add_vectors(cosine_index, xb_norm)

    return cosine_index, xb_norm, xq_norm



def build_hnsw_index(d: int, M: int, xb: np.ndarray, efConstruction=40, metric=faiss.METRIC_L2):
    """
    Build HNSW index: builds a multi-layer proximity graph where each vector
    links to M neighbours per layer, and search greedily traverses the graph
    from an entry point.

    :param d: dimension of vectors
    :param M: number of graph links per node
    :param xb: vector db
    :param efConstruction: candidate list size during graph construction
    :param metric: faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT
    :return: HNSW index
    """
    index = faiss.IndexHNSWFlat(d, M, metric)
    index.hnsw.efConstruction = efConstruction

    add_vectors(index, xb)

    return index