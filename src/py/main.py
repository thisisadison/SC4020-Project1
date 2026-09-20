from methods import *
from dataset import *
from metrics import *



def run_flat_pq_random():
    """
    Use random data for pq search
    """
    xb_random, xq_random = generate_random_data()

    flat_index = build_flat_index(xb_random, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random) # check distance between queries and vector db
    # search_index(flat_index, noNeighbors=10, noQueries=5, xq=xb_random) # check distance between vector db and itself
    measure_latency(flat_index, xq_random, k=10)
    measure_index_size(flat_index)  

    pq_index = build_pq_index(d=64, m=32, nbits=10, xb=xb_random)
    _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_random) # check distance between queries and vector db
    # search_index(pq_index, noNeighbors=10, noQueries=5, xq=xb_random) # check distance between vector db and itself
    get_metrics(pred_ids, gt_ids, pq_index, xq_random, k=10)



def run_flat_pq_clustered():
    """
    Use clustered data for pq search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    flat_index = build_flat_index(xb_clustered, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered) # check distance between queries and vector db
    # search_index(flat_index, noNeighbors=1, noQueries=5, xq=xb_clustered) # check distance between vector db and itself
    measure_latency(flat_index, xq_clustered, k=10)
    measure_index_size(flat_index)  

    pq_index = build_pq_index(d=64, m=32, nbits=10, xb=xb_clustered)
    _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_clustered) # check distance between queries and vector db
    # search_index(pq_index, noNeighbors=1, noQueries=100, xq=xb_clustered) # check distance between vector db and itself 
    get_metrics(pred_ids, gt_ids, pq_index, xq_clustered, k=10)



def run_flat_lsh_clustered():
    """
    Use clustered data for lsh search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    flat_index = build_flat_index(xb_clustered, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    measure_latency(flat_index, xq_clustered, k=10)
    measure_index_size(flat_index)

    lsh_index = build_lsh_index(d=64, nbits=256, xb=xb_clustered)
    _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    get_metrics(pred_ids, gt_ids, lsh_index, xq_clustered, k=10)



def run_cosine_lsh_clustered():
    """
    Use clustered data for lsh search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    cosine_index, xb_norm, xq_norm = build_flat_index_cosine(xb_clustered, xq_clustered, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    measure_latency(cosine_index, xq_norm, k=10)
    measure_index_size(cosine_index)

    lsh_index = build_lsh_index(d=64, nbits=256, xb=xb_norm)
    _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)

    for nbits in [512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits ==========")

        lsh_index = build_lsh_index(d=64, nbits=nbits, xb=xb_clustered)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_clustered, k=10)



def run_flat_lsh_random():
    """
    Use random data for lsh search
    """
    xb_random, xq_random = generate_random_data()

    flat_index = build_flat_index(xb_random, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random)
    measure_latency(flat_index, xq_random, k=10)
    measure_index_size(flat_index)

    lsh_index = build_lsh_index(d=64, nbits=256, xb=xb_random)
    _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_random)
    get_metrics(pred_ids, gt_ids, lsh_index, xq_random, k=10)



def run_cosine_lsh_random():
    """
    Use random data for lsh search
    """
    xb_random, xq_random = generate_random_data()

    cosine_index, xb_norm, xq_norm = build_flat_index_cosine(xb_random, xq_random, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    measure_latency(cosine_index, xq_norm, k=10)
    measure_index_size(cosine_index)

    lsh_index = build_lsh_index(d=64, nbits=256, xb=xb_norm)
    _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)

    for nbits in [512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (random data) ==========")

        lsh_index = build_lsh_index(d=64, nbits=nbits, xb=xb_random )
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_random, k=10)



def main():
    run_cosine_lsh_clustered()



if __name__ == "__main__":
    main()



"""
- increase in m generally leads to a higher recall as we have more centroids with lesser dimension per subspace.
    - at smaller vector db size or higher m, standard pq's sequential slicing breaks correlation between dimensions for clustered data, allowing uniform random data to perform better.
    - upon increasing the vector db size, more data points allow for better clustering leading to better estimation of centroids and stronger correlations within each subspace
- index size of flat index is much more significant than that of pq index due to the vector compression. computation time however is still significant due to the lookup time for centroid distance 
which introduces overhead that outweights the latency of the additional arithmetic operations of flat index linear search.
"""