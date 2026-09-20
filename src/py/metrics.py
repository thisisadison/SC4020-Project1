import pandas as pd
import numpy as np
import time
import faiss
import statistics

from dataset import generate_random_data
from methods import *



# Precision@k = (how many of your k guesses were correct) / (k, the number of guesses you made)
def precision_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    Get precision@k

    :param pred_ids: ids of predicted vectors
    :return gt_ids: ids of ground truth vectors
    :param k: number of nearest neighbors
    :return: precision@k 
    """
    print("\n=== Get Precision@k ===")
    n_queries = gt_ids.shape[0]
    precision_arr = []

    for i in range (n_queries):
        # Calculate precision for each query
        set_pred = set(pred_ids[i][:k])
        set_gt = set(gt_ids[i][:k])
        count = len(set_pred.intersection(set_gt))
        precision = count / k

        precision_arr.append(precision)

    precision = np.mean(precision_arr)
    print(f"\nprecision@{k}: {precision:.4f}")

    return precision



# Recall@k = (how many of your k guesses were correct) / (the total number of correct answers that exist)
def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    Get recall@k
    
    :param pred_ids: ids of predicted vectors
    :return gt_ids: ids of ground truth vectors
    :param k: number of nearest neighbors
    :return: recall@k 
    """
    print("\n=== Get Recall@k ===")
    n_queries = gt_ids.shape[0]
    recall_arr = []

    for i in range (n_queries):
        # Calculate recall for each query
        set_pred = set(pred_ids[i][:k])
        set_gt = set(gt_ids[i][:k])
        count = len(set_pred.intersection(set_gt))
        recall = count / k

        recall_arr.append(recall)

    recall = np.mean(recall_arr)
    print(f"\nrecall@{k}: {recall:.4f}")

    return recall



def measure_latency(index, xq: np.ndarray, k: int, n_repeats=10)-> float:
    """
    Measure latency of FAISS index search

    :param index: FAISS index
    :param xq: query dimensions, np array
    :param k: number of nearest neighbors
    :param n_repeats: number of repeats for averaging
    :return: average latency in seconds
    """
    print("\n=== Get Latency ===")
    time_list = []
    for _ in range(n_repeats):
        start_time = time.perf_counter() # record start time
        index.search(xq, k) # search k nearest neighbors for xq
        end_time = time.perf_counter() # record end time
        time_list.append(end_time - start_time)

    median_val = statistics.median(time_list) # get the median time 

    print(f"\nMedian latency for {n_repeats} repeats: {median_val:.6f} seconds")

    return median_val



def measure_index_size(index) -> int:
    """
    Measure size of FAISS index in bytes

    :param index: FAISS index
    :return: size of index in bytes
    """
    print("\n=== Get Index Size ===")
    serialized_index = faiss.serialize_index(index) # serialize index to numpy byte array
    size_in_bytes = serialized_index.nbytes
    size_in_mb = size_in_bytes / (1024 * 1024)

    print("\nIndex size (MB):", size_in_mb)  

    return size_in_mb



def get_metrics(pred_ids: np.ndarray, gt_ids: np.ndarray, index, xq: np.ndarray, k: int, n_repeats=10):
    """
    Print metrics for search and index performance
    """
    recall_at_k(pred_ids, gt_ids, k)
    measure_latency(index, xq, k, n_repeats)
    measure_index_size(index)



def test_metrics():
    xb, xq = generate_random_data(nb=1000, nq=100, d=64)
    dim = xb.shape[1]

    index = build_flat_index(dim)
    add_vectors(index, xb)

    k = 10 # top k entries
    _, gt_ids = search_index(index, noNeighbors=k, noQueries=len(xq), xq=xq)
    _, pred_ids = search_index(index, noNeighbors=k, noQueries=len(xq), xq=xq)

    measure_latency(index, xq, k)
    precision_at_k(pred_ids, gt_ids, k)
    recall_at_k(pred_ids, gt_ids, k)
    measure_index_size(index)



# def main():
#     test_metrics()

# if __name__ == "__main__":
#     main()