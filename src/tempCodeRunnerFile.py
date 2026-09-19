

"""
- increase in m generally leads to a higher recall as we have more centroids with lesser dimension per subspace.
    - at smaller vector db size or higher m, standard pq's sequential slicing breaks correlation between dimensions for clustered data, allowing uniform random data to perform better.
    - upon increasing the vector db size, more data points allow for better clustering leading to better estimation of centroids and stronger correlations within each subspace
- index size of flat index is much more significant than that of pq index due to the vector compression. computation time however is still significant due to the lookup time for centroid distance 
which introduces overhead that outweights the latency of the additional arithmetic operations of flat index linear search.
"""