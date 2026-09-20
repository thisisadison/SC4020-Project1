# Project Notes — Similarity Search

## Config reference (from current code)

### Datasets (`dataset.py`)
- `generate_random_data(nb=150000, nq=100, d=64)` — Gaussian random, normalized to unit L2 norm
- `generate_clustered_data(nb=128000, nq=100, centers=6, n_features=64, random_state=42)` — `make_blobs`, then a random linear transform stretches the spheres into correlated ellipses before splitting into `xb`/`xq`

### Methods (`methods.py`)
- `build_flat_index(xb, d)` — exact search, L2, ground truth
- `build_pq_index(d, m, nbits, xb)` — Product Quantization
- `build_opq_index(d, m, nbits, xb)` — Optimized PQ (learns a rotation before quantizing)
- `build_lsh_index(d, nbits, xb)` — random-hyperplane LSH
- `build_flat_index_cosine(xb, xq, d, k=10)` — normalized Flat + inner product, cosine-similarity ground truth

---

## PQ

### Run: `run_flat_pq_random()` — random data, m=32, nbits=9, k=10
| Metric | Flat | PQ |
|---|---|---|
| recall@10 | 1.0 | |
| latency (median, 10 repeats, s) | | |
| index size (MB) | | |

### Run: `run_flat_pq_clustered()` — clustered+correlated data, m=32, nbits=10, k=10
| Metric | Flat | PQ |
|---|---|---|
| recall@10 | 1.0 | 0.8390 |
| latency (median, 10 repeats, s) | 0.032124 | 0.323472 |
| index size (MB) | 31.25004291534424 | 5.132894515991211 |

### PQ — observations (across all runs above)
- index size of PQ is much lower compared to Flat search due to vector compression, allowing each vector to be represented by compact centroid indices rather than their original floating point values. 
- latency of PQ however is significantly higher due to the multitude of operations involved in PQ. compute heavy operations include computing distance from query subvector to centroids, constructing distance lookup tables, and accumulating distances across subspaces.
- recall using clustered data is generally lower than that of random data. clustered data has more obvious structure creating stronger correlations between certain dimensions. since sequential slicing is performed, mcorrelated dimensions may be placed into different subspaces and their cross-dimensional relationships are not directly modeled, resulting in greater quantization error and potentially poorer nearest-neighbour recall.

---

## LSH

### Run: `run_flat_lsh_clustered()` — clustered+correlated data, nbits=256, k=10
| Metric | Flat | LSH |
|---|---|---|
| recall@10 | 1.0 | 0.0460 |
| latency (median, 10 repeats, s) | 0.066568 | 0.010903 |
| index size (MB) | 31.25004291534424 | 3.968838691711426 |

### Run: `run_cosine_lsh_clustered()` — same data, cosine ground truth instead of L2, nbits=256, k=10
**Purpose:** test whether LSH's low recall (above) is a metric mismatch (LSH targets cosine similarity, not L2)

| Metric | Cosine Flat | LSH |
|---|---|---|
| recall@10 | 1.0 | 0.0500 |
| latency (median, 10 repeats, s) | 0.082911 | 0.014785 |
| index size (MB) | 31.25004291534424 | 3.968838691711426 |

**Result:** unchanged from the L2 ground truth result — confirms the metric-mismatch hypothesis was wrong (see PQ/LSH observations below).

### LSH nbits sensitivity sweep (cosine ground truth, k=10, same data)
| nbits | recall@10 | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|
| 256 | 0.0500 | 0.014672 | 3.968838691711426 |
| 512 | 0.1160 | 0.014410 | 7.937588691711426 |
| 1024 | 0.2140 | 0.045902 | 15.875088691711426 |
| 2048 | 0.3420 | 0.090716 | 31.750088691711426 |
| 4096 | 0.5120 | 0.224803 | 63.500088691711426 |


### LSH — observations (across all runs above)
- LSH on clustered/correlated data achieves substantially lower latency and index size than exact Flat search, but with significantly lower recall at 256 bits. Increasing nbits improves recall consistently, but increases both index size and search latency. 
- LSH performs worse on the clustered/correlated dataset than on the random dataset across all tested nbits values. Switching from L2 to cosine ground truth produces negligible changes in recall, suggesting that the low recall is not primarily due to metric mismatch.
- this implies that while lsh is a data independent method as their hash functions are randomly generated,their retrieval performance may not be similarly independent.

---

## LSH — random data

### Run: `run_flat_lsh_random()` — random (pre-normalized) data, nbits=256, k=10
| Metric | Flat | LSH |
|---|---|---|
| recall@10 | 1.0 | 0.2170 |
| latency (median, 10 repeats, s) | 0.081240 | 0.013934 |
| index size (MB) | 36.62113666534424 | 4.640225410461426 |

### Run: `run_flat_lsh_random()` / `run_cosine_lsh_random()` — random (pre-normalized) data, nbits=256, k=10
| Metric | Cosine Flat | LSH |
|---|---|---|
| recall@10 | 1.0 | 0.2170 |
| latency (median, 10 repeats, s) | 0.093046 | 0.017274 |
| index size (MB) | 36.62113666534424 | 4.640225410461426 |

### LSH nbits sensitivity sweep — random data, k=10
| nbits | recall@10 | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|
| 256 | 0.2170 | 0.015804 | 4.640225410461426 |
| 512 | 0.3710 | 0.018039 | 9.280362129211426 |
| 1024 | 0.5370 | 0.049788 | 18.560635566711426 |
| 2048 | 0.6660 | 0.072901 | 37.121182441711426 |
| 4096 | 0.7270 | 0.194392 | 74.24227619171143 |


### LSH Random — observations (across all runs above)
- LSH on random data achieves substantially lower latency and index size than exact Flat search at low nbits, while providing higher recall than LSH on clustered/correlated data. 
- Increasing nbits consistently improves recall, reaching 0.727 at 4096 bits, but this comes with substantial increases in index size and search latency. The results demonstrate a clear recall–efficiency trade-off.

---

## Open questions / to revisit
-
