import pandas as pd
import numpy as np

from dataset import generate_fake_data
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
    n_queries = gt_ids.shape[0]
    precision_arr = np.array([])

    for i in range (n_queries):
        # Calculate precision for each query
        set_pred = set(pred_ids[i][:k])
        set_gt = set(gt_ids[i][:k])
        count = len(set_pred.intersection(set_gt))
        precision = count / k

        np.append(precision_arr, precision)

    return np.mean(precision_arr)

# Recall@k = (how many of your k guesses were correct) / (the total number of correct answers that exist)
def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    Get recall@k
    
    :param pred_ids: ids of predicted vectors
    :return gt_ids: ids of ground truth vectors
    :param k: number of nearest neighbors
    :return: recall@k 
    """
    n_queries = gt_ids.shape[0]
    recall_arr = np.array([])

    for i in range (n_queries):
        # Calculate recall for each query
        set_pred = set(pred_ids[i][:k])
        set_gt = set(gt_ids[i][:k])
        count = len(set_pred.intersection(set_gt))
        recall = count / k

        np.append(recall_arr, recall)

    return np.mean(recall_arr)

def test_metrics():
    xb, xq = generate_fake_data(nb=1000, nq=100, d=64)
    dim = xb.shape[1]

    index = build_index(dim)
    add_vectors(index, xb)

    k = 10 # top k entries
    _, gt_ids = search_index(index, noNeighbors=k, noQueries=len(xq), xq=xq)
    _, pred_ids = search_index(index, noNeighbors=k, noQueries=len(xq), xq=xq)

    print("precision@k:", precision_at_k(pred_ids, gt_ids, k))
    print("recall@k:", recall_at_k(pred_ids, gt_ids, k))

def main():
    test_metrics()

if __name__ == "__main__":
    main()