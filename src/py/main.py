from methods import *
from dataset import *
from metrics import *



D = 384                # vector dimension for every dataset, set by the all-MiniLM-L6-v2 embedding size
SYNTHETIC_NB = 100000  # synthetic vector db size
SYNTHETIC_NQ = 100     # synthetic query count



def run_lsh_cosine_clustered():
    """
    Use clustered data for lsh search
    """
    xb_clustered, xq_clustered = generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D)

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_clustered, xq=xq_clustered, d=D)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (clustered data, cosine ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=D, nbits=nbits, xb=xb_norm)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)



def run_lsh_cosine_random():
    """
    Use random data for lsh search
    """
    xb_random, xq_random = generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D)

    print(f"\n========== Cosine Flat (random data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_random, xq=xq_random, d=D)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (random data, cosine ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=D, nbits=nbits, xb=xb_norm)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)



def run_pq_cosine_random():
    """
    Use random data for pq search with cosine ground truth
    """
    xb_random, xq_random = generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D)

    print(f"\n========== Cosine Flat (random data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_random, xq=xq_random, d=D)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (random data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=32, nbits=nbits, xb=xb_norm, metric=faiss.METRIC_L2)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)

    for m in [8, 16, 64, 192]:

        print(f"\n========== PQ m={m}, nbits=10 (random data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=m, nbits=10, xb=xb_norm, metric=faiss.METRIC_L2)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)



def run_pq_cosine_clustered():
    """
    Use clustered data for pq search with cosine ground truth
    """
    xb_clustered, xq_clustered = generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D)

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_clustered, xq=xq_clustered, d=D)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (clustered data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=32, nbits=nbits, xb=xb_norm, metric=faiss.METRIC_L2)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)

    for m in [8, 16, 64, 192]:

        print(f"\n========== PQ m={m}, nbits=10 (clustered data, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=D, m=m, nbits=10, xb=xb_norm, metric=faiss.METRIC_L2)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)



def run_hnsw_cosine_random():
    """
    Use random data for hnsw search with cosine ground truth
    """
    xb_random, xq_random = generate_random_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, d=D)

    print(f"\n========== Cosine Flat (random data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_random, xq=xq_random, d=D)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    hnsw_index = measure_build_time(build_hnsw_index, d=D, M=32, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)

    for efSearch in [16, 32, 64, 128, 256]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (random data, cosine ground truth) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)

    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (random data, cosine ground truth) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=D, M=M, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)



def run_hnsw_cosine_clustered():
    """
    Use clustered data for hnsw search with cosine ground truth
    """
    xb_clustered, xq_clustered = generate_clustered_data(nb=SYNTHETIC_NB, nq=SYNTHETIC_NQ, n_features=D)

    print(f"\n========== Cosine Flat (clustered data) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_clustered, xq=xq_clustered, d=D)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=100, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    hnsw_index = measure_build_time(build_hnsw_index, d=D, M=32, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)

    for efSearch in [16, 32, 64, 128, 256]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (clustered data, cosine ground truth) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)

    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (clustered data, cosine ground truth) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=D, M=M, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=100, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)



def run_pq_cosine_finance():
    """
    Use financebench filing chunks for pq search with cosine ground truth
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    d = xb_fin.shape[1]  # 384 for all-MiniLM-L6-v2
    nq = xq_fin.shape[0]  # all 150 questions

    print(f"\n========== Cosine Flat (financebench) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_fin, xq=xq_fin, d=d)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [6, 8, 10]:

        print(f"\n========== PQ m=32, nbits={nbits} (financebench, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=d, m=32, nbits=nbits, xb=xb_norm, metric=faiss.METRIC_L2)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)

    for m in [8, 16, 64, 192]:

        print(f"\n========== PQ m={m}, nbits=10 (financebench, cosine ground truth) ==========")

        pq_index = measure_build_time(build_pq_index, d=d, m=m, nbits=10, xb=xb_norm, metric=faiss.METRIC_L2)
        _, pred_ids = search_index(pq_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, pq_index, xq_norm, k=10)



def run_lsh_cosine_finance():
    """
    Use financebench filing chunks for lsh search with cosine ground truth
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    d = xb_fin.shape[1]  # 384 for all-MiniLM-L6-v2
    nq = xq_fin.shape[0]  # all 150 questions

    print(f"\n========== Cosine Flat (financebench) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_fin, xq=xq_fin, d=d)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    for nbits in [256, 512, 1024, 2048, 4096]:

        print(f"\n========== LSH {nbits} bits (financebench, cosine ground truth) ==========")

        lsh_index = measure_build_time(build_lsh_index, d=d, nbits=nbits, xb=xb_norm)
        _, pred_ids = search_index(lsh_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, lsh_index, xq_norm, k=10)



def run_hnsw_cosine_finance():
    """
    Use financebench filing chunks for hnsw search with cosine ground truth
    """
    xb_fin, xq_fin, _ = generate_financebench_data()
    d = xb_fin.shape[1]  # 384 for all-MiniLM-L6-v2
    nq = xq_fin.shape[0]  # all 150 questions

    print(f"\n========== Cosine Flat (financebench) ==========")

    cosine_index, xb_norm, xq_norm = measure_build_time(build_flat_index_cosine, xb=xb_fin, xq=xq_fin, d=d)
    _, gt_ids = search_index(cosine_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
    flat_latency = measure_latency(cosine_index, xq_norm, k=10)
    flat_size = measure_index_size(cosine_index)

    print(f"recall@10: 1.0000 (ground truth) | latency: {flat_latency:.6f}s | index size: {flat_size:.4f}MB\n")

    hnsw_index = measure_build_time(build_hnsw_index, d=d, M=32, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)

    for efSearch in [16, 32, 64, 128, 256]:

        print(f"\n========== HNSW M=32, efSearch={efSearch} (financebench, cosine ground truth) ==========")

        hnsw_index.hnsw.efSearch = efSearch
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)

    for M in [8, 16, 64]:

        print(f"\n========== HNSW M={M}, efSearch=64 (financebench, cosine ground truth) ==========")

        hnsw_index = measure_build_time(build_hnsw_index, d=d, M=M, xb=xb_norm, metric=faiss.METRIC_INNER_PRODUCT)
        hnsw_index.hnsw.efSearch = 64
        _, pred_ids = search_index(hnsw_index, noNeighbors=10, noQueries=nq, xq=xq_norm)
        get_metrics(pred_ids, gt_ids, hnsw_index, xq_norm, k=10)



def run_all_synthetic():
    run_lsh_cosine_random()
    run_lsh_cosine_clustered()

    run_pq_cosine_random()
    run_pq_cosine_clustered()

    run_hnsw_cosine_random()
    run_hnsw_cosine_clustered()



def run_all_finance():
    run_lsh_cosine_finance()
    run_pq_cosine_finance()
    run_hnsw_cosine_finance()



def main():
    run_all_synthetic()
    run_all_finance()



if __name__ == "__main__":
    main()