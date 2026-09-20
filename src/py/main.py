from methods import *
from dataset import *
from metrics import *



def run_pq_l2_random():
    """
    Use random data for pq search
    """
    xb_random, xq_random = generate_random_data()

    print(f"\n========== Flat (random data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_random, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random) # check distance between queries and vector db
    # search_index(flat_index, noNeighbors=10, noQueries=5, xq=xb_random) # check distance between vector db and itself
    flat_latency = measure_latency(flat_index, xq_random, k=10)
    flat_size = measure_index_size(flat_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (random data) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=32, nbits=nbits, xb=xb_random)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_random) # check distance between queries and vector db
        # search_index(pq_index, noNeighbors=10, noQueries=5, xq=xb_random) # check distance between vector db and itself
        get_metrics(pred_ids, gt_ids, pq_index, xq_random, k=10)

    # m=32 at nbits=10 already measured in the loop above
    for m in [4, 8, 16]:

        print(f"\n========== PQ m={m}, nbits=10 (random data) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=m, nbits=10, xb=xb_random)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, pq_index, xq_random, k=10)



def run_pq_cosine_random():
    """
    Use random data for pq search with cosine ground truth
    """
    xb_random, xq_random = generate_random_data()

    print(f"\n========== Cosine Flat (random data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_random, xq=xq_random, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (random data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=32, nbits=nbits, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)

    # m=32 at nbits=10 already measured in the loop above
    for m in [4, 8, 16]:

        print(f"\n========== PQ m={m}, nbits=10 (random data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=m, nbits=10, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)



def run_pq_l2_clustered():
    """
    Use clustered data for pq search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    print(f"\n========== Flat (clustered data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_clustered, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered) # check distance between queries and vector db
    # search_index(flat_index, noNeighbors=1, noQueries=5, xq=xb_clustered) # check distance between vector db and itself
    flat_latency = measure_latency(flat_index, xq_clustered, k=10)
    flat_size = measure_index_size(flat_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (clustered data) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=32, nbits=nbits, xb=xb_clustered)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_clustered) # check distance between queries and vector db
        # search_index(pq_index, noNeighbors=1, noQueries=100, xq=xb_clustered) # check distance between vector db and itself 
        get_metrics(pred_ids, gt_ids, pq_index, xq_clustered, k=10)

    # m=32 at nbits=10 already measured in the loop above
    for m in [4, 8, 16]:

        print(f"\n========== PQ m={m}, nbits=10 (clustered data) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=m, nbits=10, xb=xb_clustered)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, pq_index, xq_clustered, k=10)



def run_pq_cosine_clustered():
    """
    Use clustered data for pq search with cosine ground truth
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_clustered, xq=xq_clustered, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (clustered data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=32, nbits=nbits, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)

    # m=32 at nbits=10 already measured in the loop above
    for m in [4, 8, 16]:

        print(f"\n========== PQ m={m}, nbits=10 (clustered data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=64, m=m, nbits=10, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)



def run_lsh_l2_clustered():
    """
    Use clustered data for lsh search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    print(f"\n========== Flat (clustered data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_clustered, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    flat_latency = measure_latency(flat_index, xq_clustered, k=10)
    flat_size = measure_index_size(flat_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (clustered data, l2 ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=64, nbits=nbits, xb=xb_clustered)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_clustered, k=10)



def run_lsh_cosine_clustered():
    """
    Use clustered data for lsh search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_clustered, xq=xq_clustered, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (clustered data, cosine ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=64, nbits=nbits, xb=xb_norm)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)



def run_lsh_l2_random():
    """
    Use random data for lsh search
    """
    xb_random, xq_random = generate_random_data()

    print(f"\n========== Flat (random data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_random, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random)
    flat_latency = measure_latency(flat_index, xq_random, k=10)
    flat_size = measure_index_size(flat_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (random data, l2 ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=64, nbits=nbits, xb=xb_random)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_random, k=10)



def run_lsh_cosine_random():
    """
    Use random data for lsh search
    """
    xb_random, xq_random = generate_random_data()

    print(f"\n========== Cosine Flat (random data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_random, xq=xq_random, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (random data, cosine ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=64, nbits=nbits, xb=xb_norm)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)



def run_hnsw_cosine_random():
    """
    Use random data for hnsw search with cosine ground truth
    """
    xb_random, xq_random = generate_random_data()

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_random, xq=xq_random, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    # efSearch is a search-time knob, so one index serves the whole sweep
    hnsw_index = measure_build_time(build_hnsw_index, d=64, M=32, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)

    for efSearch in [16, 32, 64, 128]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (clustered data, cosine ground truth) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)

    # M=32 at efSearch=64 already measured in the loop above
    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (clustered data, cosine ground truth) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=64, M=M, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)



def run_hnsw_l2_random():
    """
    Use random data for hnsw search
    """
    xb_random, xq_random = generate_random_data()

    print(f"\n========== Flat (random data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_random, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_random)
    flat_latency = measure_latency(flat_index, xq_random, k=10)
    flat_size = measure_index_size(flat_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    hnsw_index = measure_build_time(build_hnsw_index, d=64, M=32, xb=xb_random)

    for efSearch in [16, 32, 64, 128]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (random data) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_random, k=10)

    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (random data) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=64, M=M, xb=xb_random)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_random)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_random, k=10)



def run_hnsw_l2_clustered():
    """
    Use clustered data for hnsw search
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    print(f"\n========== Flat (clustered data) ==========")

    flat_index = measure_build_time(build_flat_index, xb=xb_clustered, d=64)
    _, gt_ids = search_index(flat_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
    flat_latency = measure_latency(flat_index, xq_clustered, k=10)
    flat_size = measure_index_size(flat_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    hnsw_index = measure_build_time(build_hnsw_index, d=64, M=32, xb=xb_clustered)

    for efSearch in [16, 32, 64, 128]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (clustered data) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_clustered, k=10)

    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (clustered data) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=64, M=M, xb=xb_clustered)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_clustered)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_clustered, k=10)



def run_hnsw_cosine_clustered():
    """
    Use clustered data for hnsw search with cosine ground truth
    """
    xb_clustered, xq_clustered = generate_clustered_data()

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_clustered, xq=xq_clustered, d=64)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    # efSearch is a search-time knob, so one index serves the whole sweep
    hnsw_index = measure_build_time(build_hnsw_index, d=64, M=32, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)

    for efSearch in [16, 32, 64, 128]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (clustered data, cosine ground truth) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)

    # M=32 at efSearch=64 already measured in the loop above
    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (clustered data, cosine ground truth) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=64, M=M, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)



def main():
    run_hnsw_cosine_random()



if __name__ == "__main__":
    main()