import faiss
import numpy as np

from dataset import generate_fake_data

def build_index(d: int):
    """
    Build FAISS index for vector search

    :param d: dimension of vectors
    :return: vector db
    """
    index = faiss.IndexFlatL2(d);  # build index
    print("\nTrained:", index.is_trained)
    return index

def add_vectors(index, xb: np.ndarray):
    """
    Populate FAISS index with vectors

    :param index: FAISS index
    :param xb: vector db dimensions, np array
    """
    index.add(xb)
    print(f"\nxb = {index.ntotal} vectors added to index")

def search_index(index, noNeighbors: int, noQueries: int, xq: np.ndarray) -> tuple[np.ndarray: np.ndarray]:
    """
    Search FAISS index for nearest neighbors

    :param index: FAISS index
    :param noNeighbors: no of nearest neighbors
    :param noQueries: no of queries
    :param xq: query dimensions, np array
    """
    d, i = index.search(xq[:noQueries], noNeighbors)
    print("\nDistance between query and neighbors:")
    print(d)

    print("\nIndex of neighbors:")
    print(i)

    return d, i

def main():
    print()
    xb, xq = generate_fake_data(nb=10000, nq=100, d=64)
    index = build_index(d=64)
    add_vectors(index, xb)
    search_index(index, noNeighbors=4, noQueries=5, xq=xq) # check distance between queries and vector db
    search_index(index, noNeighbors=1, noQueries=5, xq=xb) # check distance between vector db and itself

if __name__ == "__main__":
    main()

