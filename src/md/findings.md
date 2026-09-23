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

## PQ - random data

### Run: `run_pq_l2_random()` — random data, k=10

**Baseline (Flat, exact search)**
| Metric | Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.014301 |
| latency (median, 10 repeats, s) | 0.063182 |
| index size (MB) | 36.6211 |

**PQ nbits sensitivity sweep — m=32 fixed**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 6 | 0.6380 | 2.252211 | 0.368227 | 3.4489 |
| 8 | 0.7860 | 4.286942 | 0.051317 | 4.6402 |
| 10 | 0.8650 | 16.656300 | 0.346519 | 5.9721 |

**PQ m sensitivity sweep — nbits=10 fixed**
| m | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 4 | 0.0920 | 45.387165 | 0.033183 | 0.9653 |
| 8 | 0.2750 | 9.408497 | 0.094754 | 1.6806 |
| 16 | 0.5740 | 9.228064 | 0.151230 | 3.1111 |
| 32 | 0.8650 | 16.656300 | 0.346519 | 5.9721 |

### Run: `run_pq_cosine_random()` — random (pre-normalized) data, cosine ground truth, k=10

**Baseline (Cosine Flat, exact search)**
| Metric | Cosine Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.041190 |
| latency (median, 10 repeats, s) | 0.083225 |
| index size (MB) | 36.6211 |

**PQ nbits sensitivity sweep — m=32 fixed**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 6 | 0.6800 | 1.334115 | 0.260197 | 3.4489 |
| 8 | 0.8090 | 2.751381 | 0.053284 | 4.6402 |
| 10 | 0.8880 | 15.524865 | 0.334592 | 5.9721 |

**PQ m sensitivity sweep — nbits=10 fixed**
| m | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 4 | 0.0940 | 46.482731 | 0.034781 | 0.9653 |
| 8 | 0.2770 | 9.450026 | 0.078093 | 1.6806 |
| 16 | 0.6090 | 10.667675 | 0.158002 | 3.1111 |
| 32 | 0.8880 | 15.524865 | 0.334592 | 5.9721 |

---

## PQ - clustered data

### Run: `run_pq_l2_clustered()` — clustered+correlated data, k=10

**Baseline (Flat, exact search)**
| Metric | Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.017712 |
| latency (median, 10 repeats, s) | 0.079612 |
| index size (MB) | 31.2500 |

**PQ nbits sensitivity sweep — m=32 fixed**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 6 | 0.4520 | 1.412795 | 0.226414 | 2.9454 |
| 8 | 0.7070 | 2.849400 | 0.049724 | 3.9688 |
| 10 | 0.8390 | 11.601651 | 0.299648 | 5.1329 |

**PQ m sensitivity sweep — nbits=10 fixed**
| m | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 4 | 0.1270 | 35.661743 | 0.040311 | 0.8604 |
| 8 | 0.2680 | 7.495609 | 0.067623 | 1.4708 |
| 16 | 0.5490 | 8.062942 | 0.135738 | 2.6915 |
| 32 | 0.8390 | 11.601651 | 0.299648 | 5.1329 |

### Run: `run_pq_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat, exact search)**
| Metric | Cosine Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.042466 |
| latency (median, 10 repeats, s) | 0.073922 |
| index size (MB) | 31.2500 |

**PQ nbits sensitivity sweep — m=32 fixed**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 6 | 0.0290 | 1.285376 | 0.208914 | 2.9454 |
| 8 | 0.1230 | 2.575009 | 0.043865 | 3.9688 |
| 10 | 0.3520 | 14.741809 | 0.327062 | 5.1329 |

**PQ m sensitivity sweep — nbits=10 fixed**
| m | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 4 | 0.0050 | 37.340161 | 0.035240 | 0.8604 |
| 8 | 0.0100 | 8.318004 | 0.070042 | 1.4708 |
| 16 | 0.0560 | 8.687295 | 0.136507 | 2.6915 |
| 32 | 0.3520 | 14.741809 | 0.327062 | 5.1329 |

---

### PQ — observations (across both datasets above)

* **PQ substantially reduces index size compared with Flat.**

  Across the experiments, PQ indexes were much smaller than the Flat index. For example, on random data, the Flat index was **36.62 MB**, while PQ ranged from **0.97-5.97 MB**. On clustered data, PQ ranged from **0.86-5.13 MB** compared with **31.25 MB** for Flat.

* **Increasing `nbits` consistently improves recall.**

  With `m=32`, recall increased as `nbits` increased. On random L2 data, recall rose from **0.638 → 0.786 → 0.865** for 6 → 8 → 10 bits. The same trend appeared on clustered L2 data, increasing from **0.452 → 0.707 → 0.839**.

* **Increasing `m` also substantially improves recall.**

  With `nbits=10`, random L2 recall increased from **0.092 → 0.275 → 0.574 → 0.865** as `m` increased from 4 → 8 → 16 → 32. This shows that using more subspaces allowed the vectors to be represented more accurately.

* **`m` has a larger effect on recall than `nbits`.**

  Varying `m` from 4 to 32 moved random L2 recall across **0.092-0.865**, while varying `nbits` from 6 to 10 moved it only across **0.638-0.865**. Losing subspaces discards more information than using coarser centroids within each subspace.

* **Higher recall comes at the cost of larger indexes.**

  On clustered L2 data, increasing `m` from 4 to 32 increased the index size from **0.86 MB to 5.13 MB**, while recall increased from **0.127 to 0.839**. Similarly, increasing `nbits` from 6 to 10 increased the index size from **2.95 MB to 5.13 MB**, while recall increased from **0.452 to 0.839**.

* **`nbits=8` is markedly faster than its neighbours.**

  At `m=32` on random L2 data, latency was **0.051s at `nbits=8`** compared with **0.368s at 6 bits** and **0.347s at 10 bits**. The same pattern appeared on clustered L2 data (**0.050s** against **0.226s** and **0.300s**). At 8 bits each code occupies exactly one byte, so the lookup is byte-aligned; 6 and 10 bits require unpacking codes that straddle byte boundaries. The best speed/accuracy operating point for PQ here is therefore the byte-aligned one rather than the highest-resolution one.

* **PQ build time is much higher than Flat.**

  For example, on clustered L2 data, Flat took only **0.018s** to build, while PQ with `m=32, nbits=10` took **11.60s**. This is the main computational cost of PQ in these experiments.

* **Build time is driven by the number of dimensions per subspace, not the number of subspaces.**

  The slowest build in the whole study was `m=4, nbits=10` at **45.39s** on random data, which also produced the worst recall (**0.092**). With `m=4` each subspace has 16 dimensions, and fitting 1024 centroids in 16 dimensions is far harder than in the 2 dimensions used when `m=32`.

* **PQ recall was lower on clustered data than random data for L2.**

  At `m=32, nbits=10`, recall was **0.865 on random data** compared with **0.839 on clustered data**. The difference is relatively small, but the clustered dataset was slightly harder for this configuration.

* **The clustered penalty shrinks as resolution increases, and reverses at low `m`.**

  At `m=32` the random-minus-clustered gap narrowed from **-0.186** at `nbits=6` to **-0.079** at 8 bits and **-0.026** at 10 bits. Sequential slicing splits correlated dimensions across subspaces, and additional centroids absorb part of that error. At `m=4` the ordering reversed, with clustered data scoring **0.127** against **0.092** for random: with 16-dimensional subspaces, cluster structure gives k-means something to fit, whereas isotropic Gaussian data offers none.

* **Inner product outperformed L2 on the pre-normalized random dataset.**

  Because that data is already unit-norm, the L2 and cosine ground truths select the same neighbours, so only the search metric differs. Inner product won at all six configurations, by **+0.002 to +0.042**. Quantization moves each codeword off the unit sphere, and the L2 objective penalises that displacement while a cosine ground truth does not.

* **Cosine search showed a much larger drop in PQ recall on clustered data.**

  At `m=32, nbits=10`, clustered-data recall was **0.839 for L2 but only 0.352 for cosine**. Unlike the random dataset, the clustered data is not unit-norm, so normalizing changes the vectors and therefore changes which neighbours are correct. This is a different retrieval task rather than the same task measured differently, and normalizing discards magnitude information that carried much of the cluster separation.

---

## LSH — random data

### Run: `run_lsh_l2_random()` — random (pre-normalized) data, L2 ground truth, k=10

**Baseline (Flat, exact search)**
| Metric | Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.023276 |
| latency (median, 10 repeats, s) | 0.146238 |
| index size (MB) | 36.6211 |

**LSH nbits sensitivity sweep**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 256 | 0.2170 | 0.303144 | 0.017394 | 4.6402 |
| 512 | 0.3710 | 0.240314 | 0.019676 | 9.2804 |
| 1024 | 0.5370 | 0.538945 | 0.044032 | 18.5606 |
| 2048 | 0.6660 | 2.159177 | 0.046255 | 37.1212 |
| 4096 | 0.7270 | 7.564071 | 0.198563 | 74.2423 |

### Run: `run_lsh_cosine_random()` — random (pre-normalized) data, cosine ground truth, k=10

**Baseline (Cosine Flat, exact search)**
| Metric | Cosine Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.051388 |
| latency (median, 10 repeats, s) | 0.099970 |
| index size (MB) | 36.6211 |

**LSH nbits sensitivity sweep**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 256 | 0.2170 | 0.118306 | 0.019770 | 4.6402 |
| 512 | 0.3710 | 0.184638 | 0.032814 | 9.2804 |
| 1024 | 0.5370 | 0.979010 | 0.059998 | 18.5606 |
| 2048 | 0.6660 | 0.959672 | 0.097421 | 37.1212 |
| 4096 | 0.7270 | 3.334273 | 0.186199 | 74.2423 |

---

### LSH random — observations (across both runs above)

* **LSH recall increases as the number of hash bits increases.**

  On random data, recall increased from **0.217 → 0.371 → 0.537 → 0.666 → 0.727** as `nbits` increased from 256 → 512 → 1024 → 2048 → 4096. The improvement is steady rather than sudden, because Hamming distance estimates the angle between vectors with a variance that falls roughly as 1/`nbits`.

* **Increasing `nbits` increases index size substantially.**

  The index grew from **4.64 MB at 256 bits** to **74.24 MB at 4096 bits**. Therefore, improving recall required a significant increase in storage. At 4096 bits the index is roughly twice the size of the uncompressed Flat index (**36.62 MB**) while still only reaching **0.727** recall.

* **LSH has relatively low build time.**

  Build time remained low at smaller numbers of bits: the 256-bit index took only **0.30s** to build for L2 and **0.12s** for cosine. Build time increased as more hash bits were added, reaching **7.56s** for 4096-bit L2 and **3.33s** for 4096-bit cosine.

* **L2 and cosine produced identical recall on the random dataset.**

  Recall was identical for every `nbits` value in the L2 and cosine experiments. This is consistent with the random dataset being pre-normalized, where L2 distance and cosine similarity produce the same nearest-neighbour ranking: for unit vectors, squared L2 distance equals 2 - 2 × (cosine similarity), so the two orderings cannot differ. A further reason the predictions are unchanged is that random-hyperplane hashing passes through the origin, so scaling a vector cannot change the sign of any projection and therefore cannot change its hash code.

---

## LSH - clustered data

### Run: `run_lsh_l2_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat, exact search)**
| Metric | Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.013612 |
| latency (median, 10 repeats, s) | 0.074882 |
| index size (MB) | 31.2500 |

**LSH nbits sensitivity sweep**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 256 | 0.0460 | 0.087027 | 0.013785 | 3.9688 |
| 512 | 0.1090 | 0.102341 | 0.013195 | 7.9376 |
| 1024 | 0.2010 | 0.319883 | 0.042372 | 15.8751 |
| 2048 | 0.2950 | 0.778814 | 0.068033 | 31.7501 |
| 4096 | 0.4570 | 3.979888 | 0.166851 | 63.5001 |

### Run: `run_lsh_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat, exact search)**
| Metric | Cosine Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.054396 |
| latency (median, 10 repeats, s) | 0.078370 |
| index size (MB) | 31.2500 |

**LSH nbits sensitivity sweep**
| nbits | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 256 | 0.0500 | 0.086661 | 0.010880 | 3.9688 |
| 512 | 0.1160 | 0.091588 | 0.014136 | 7.9376 |
| 1024 | 0.2140 | 0.348371 | 0.044977 | 15.8751 |
| 2048 | 0.3420 | 0.798488 | 0.068373 | 31.7501 |
| 4096 | 0.5120 | 2.920417 | 0.177258 | 63.5001 |

---

### LSH clustered — observations (across both runs above)

* **LSH recall was substantially lower on clustered data than random data.**

  At 256 bits, recall was only **0.046 on clustered data**, compared with **0.217 on random data**. At 4096 bits, recall reached **0.457 on clustered data**, compared with **0.727 on random data**.

* **Increasing `nbits` still improves recall, but the improvement is limited.**

  On clustered L2 data, recall increased from **0.046 → 0.457** as `nbits` increased from 256 → 4096. However, even with 4096 bits, recall remained below the random-data result of **0.727**.

* **Higher recall requires substantially more storage.**

  The clustered L2 index increased from **3.97 MB at 256 bits** to **63.50 MB at 4096 bits**, while recall increased from **0.046 to 0.457**. The index exceeds the exact Flat index (**31.25 MB**) at 2048 bits, where recall is still only **0.295**: beyond that point LSH costs more than storing the raw vectors while returning worse answers.

* **The clustered dataset was more difficult for LSH than the random dataset.**

  The lower recall across all tested bit counts shows that the random-hyperplane hashing used here preserved the nearest-neighbour relationships less effectively for the clustered and correlated data. The random dataset is spread isotropically over the unit sphere, so true neighbours are separated by comparatively large angles that a random hyperplane is unlikely to split. Clustering concentrates vectors into narrow angular regions, which compresses exactly the signal the hashing depends on.

* **Retrieval quality is not data-independent even though the hash functions are.**

  The hyperplanes are generated without reference to the data, but the results above show performance varying by a factor of two to five between datasets. Data-independence describes how the index is constructed, not how well it performs.

* **Changing the ground truth from L2 to cosine produced a small but consistent improvement.**

  Recall rose at every bit count, from **+0.004** at 256 bits to **+0.055** at 4096 bits. Unlike the random dataset, the clustered data is not unit-norm, so the two ground truths genuinely differ here. The effect confirms that LSH tracks angular rather than Euclidean similarity, but it is far too small to account for recall as low as **0.046**, so metric mismatch is not the explanation for the poor clustered results.

---

## HNSW - random data

### Run: `run_hnsw_l2_random()` — random (pre-normalized) data, L2 ground truth, k=10

**Baseline (Flat, exact search)**
| Metric | Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.017756 |
| latency (median, 10 repeats, s) | 0.086714 |
| index size (MB) | 36.6211 |

**HNSW efSearch sensitivity sweep — M=32 fixed (build time 5.873030s, one index reused)**
| efSearch | recall@10 | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|
| 16 | 0.2570 | 0.002141 | 75.5617 |
| 32 | 0.4240 | 0.004043 | 75.5617 |
| 64 | 0.5890 | 0.005679 | 75.5617 |
| 128 | 0.7760 | 0.008615 | 75.5617 |

**HNSW M sensitivity sweep — efSearch=64 fixed**
| M | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 8 | 0.1590 | 3.110647 | 0.003744 | 48.1535 |
| 16 | 0.3500 | 4.362154 | 0.004141 | 57.2683 |
| 32 | 0.5890 | 5.873030 | 0.005679 | 75.5617 |
| 64 | 0.6600 | 7.990787 | 0.016377 | 112.1653 |

### Run: `run_hnsw_cosine_random()` — random (pre-normalized) data, cosine ground truth, k=10

**Baseline (Cosine Flat, exact search)**
| Metric | Cosine Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.054714 |
| latency (median, 10 repeats, s) | 0.109233 |
| index size (MB) | 36.6211 |

**HNSW efSearch sensitivity sweep — M=32 fixed (build time 7.246802s, one index reused)**
| efSearch | recall@10 | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|
| 16 | 0.2730 | 0.001706 | 75.5617 |
| 32 | 0.4140 | 0.003040 | 75.5617 |
| 64 | 0.5900 | 0.004561 | 75.5617 |
| 128 | 0.7520 | 0.007753 | 75.5617 |

**HNSW M sensitivity sweep — efSearch=64 fixed**
| M | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 8 | 0.1470 | 2.717913 | 0.002364 | 48.1535 |
| 16 | 0.3410 | 3.968604 | 0.008706 | 57.2683 |
| 32 | 0.5900 | 7.246802 | 0.004561 | 75.5617 |
| 64 | 0.6700 | 9.258889 | 0.009552 | 112.1653 |

---

### HNSW random — observations (across both runs above)

* **Increasing `efSearch` consistently improves recall.**

  With `M=32`, random L2 recall increased from **0.257 → 0.424 → 0.589 → 0.776** as `efSearch` increased from 16 → 32 → 64 → 128. Cosine showed a similar trend, increasing from **0.273 → 0.414 → 0.590 → 0.752**.

* **`efSearch` is a search-time parameter, so accuracy can be tuned without rebuilding.**

  A single `M=32` index spanned recall **0.257 to 0.776** with no rebuild. Neither PQ nor LSH offers this: every accuracy change in those methods requires constructing a new index.

* **Higher `efSearch` increases search latency.**

  For random L2 data, latency increased from **0.0021s at `efSearch=16`** to **0.0086s at `efSearch=128`**. Thus, higher recall required more search time.

* **Increasing `M` improves recall but increases index size and build time.**

  With `efSearch=64`, random L2 recall increased from **0.159 at `M=8`** to **0.660 at `M=64`**. At the same time, index size increased from **48.15 MB to 112.17 MB**, while build time increased from **3.11s to 7.99s**.

* **HNSW was much larger than Flat in these experiments.**

  The Flat index was **36.62 MB**, while HNSW ranged from **48.15 MB to 112.17 MB** depending on `M`. HNSW stores the original vectors uncompressed and adds the graph on top, so it can never be smaller than Flat. The overhead is close to 2 × `M` × 4 bytes per vector: at `M=64` the excess is 112.17 - 36.62 = **75.54 MB** over 150,000 vectors, or **528 bytes per vector**, against 512 bytes for the layer-0 links.

* **Recall on the random dataset had not saturated at the largest `efSearch` tested.**

  The final step from 64 to 128 still added **+0.187** recall, the largest single increase in the sweep. The values reported here therefore understate what HNSW can reach on this dataset, and extending the sweep beyond 128 would be needed for a fair ceiling.

* **L2 and cosine were effectively identical on the random dataset.**

  All seven configurations matched to within **±0.024**. The dataset is pre-normalized, so both ground truths select the same neighbours and the small differences reflect the approximate nature of graph traversal rather than any metric effect.

---

## HNSW - clustered data

### Run: `run_hnsw_l2_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat, exact search)**
| Metric | Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.013697 |
| latency (median, 10 repeats, s) | 0.050028 |
| index size (MB) | 31.2500 |

**HNSW efSearch sensitivity sweep — M=32 fixed (build time 4.065341s, one index reused)**
| efSearch | recall@10 | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|
| 16 | 0.8010 | 0.001261 | 64.4790 |
| 32 | 0.9330 | 0.001826 | 64.4790 |
| 64 | 0.9860 | 0.003218 | 64.4790 |
| 128 | 0.9980 | 0.005407 | 64.4790 |

**HNSW M sensitivity sweep — efSearch=64 fixed**
| M | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 8 | 0.7470 | 1.626998 | 0.001451 | 41.0897 |
| 16 | 0.9020 | 1.988759 | 0.002731 | 48.8696 |
| 32 | 0.9860 | 4.065341 | 0.003218 | 64.4790 |
| 64 | 0.9920 | 4.318806 | 0.004175 | 95.7141 |

### Run: `run_hnsw_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat, exact search)**
| Metric | Cosine Flat |
|---|---|
| recall@10 | 1.0 (ground truth) |
| build time (s) | 0.031841 |
| latency (median, 10 repeats, s) | 0.079028 |
| index size (MB) | 31.2500 |

**HNSW efSearch sensitivity sweep — M=32 fixed (build time 4.062337s, one index reused)**
| efSearch | recall@10 | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|
| 16 | 0.8030 | 0.001146 | 64.4790 |
| 32 | 0.9320 | 0.002512 | 64.4790 |
| 64 | 0.9810 | 0.003094 | 64.4790 |
| 128 | 0.9970 | 0.005693 | 64.4790 |

**HNSW M sensitivity sweep — efSearch=64 fixed**
| M | recall@10 | build time (s) | latency (median, 10 repeats, s) | index size (MB) |
|---|---|---|---|---|
| 8 | 0.7350 | 1.494969 | 0.001746 | 41.0897 |
| 16 | 0.8960 | 1.876298 | 0.002713 | 48.8696 |
| 32 | 0.9810 | 4.062337 | 0.003094 | 64.4790 |
| 64 | 0.9940 | 3.924521 | 0.019436 | 95.7141 |

---

### HNSW clustered — observations (across both runs above)

* **HNSW performed substantially better on clustered data than random data.**

  With `M=32`, increasing `efSearch` from 16 to 128 increased clustered L2 recall from **0.801 → 0.933 → 0.986 → 0.998**. This was much higher than the corresponding random-data recalls of **0.257 → 0.424 → 0.589 → 0.776**. Greedy graph traversal needs a distance gradient to follow; cluster structure supplies one, whereas isotropic high-dimensional data suffers from distance concentration and leaves the search with little to descend.

* **Clustered data achieved very high recall with relatively low latency.**

  At `M=32, efSearch=64`, clustered L2 achieved **0.986 recall** with a median latency of only **0.0032s**. At `efSearch=128`, recall increased further to **0.998** with **0.0054s** latency. Flat search on the same data required **0.050s**, so HNSW was about **15x faster** while returning almost all of the correct neighbours.

* **Increasing `M` improves recall, but with additional storage and build cost.**

  At `efSearch=64`, clustered L2 recall increased from **0.747 at `M=8`** to **0.992 at `M=64`**. Index size increased from **41.09 MB to 95.71 MB**.

* **`M=32` is the point of diminishing returns on this dataset.**

  Moving from `M=32` to `M=64` added only **+0.006** recall (0.986 to 0.992) while increasing the index from **64.48 MB to 95.71 MB**, a 48% storage increase for a negligible accuracy gain.

* **The effect of `M` was particularly strong on clustered data.**

  Increasing `M` from 8 to 32 increased clustered L2 recall from **0.747 to 0.986**, while the same change on random L2 data increased recall from **0.159 to 0.589**.

* **Normalizing the clustered data barely affected HNSW, unlike PQ.**

  Applying the same normalization that reduced PQ recall from **0.839 to 0.352** (a 58% drop) changed HNSW recall only from **0.986 to 0.981** (0.5%), and all seven configurations agreed to within **±0.012**. Both methods searched the same vectors against the same ground truth, and the only structural difference is that PQ stores compressed codes while HNSW stores the vectors uncompressed. This isolates quantization error, rather than any loss of neighbour structure in the data, as the cause of the PQ result.

---

## Cross-method comparison

* **PQ achieved higher recall at comparable index sizes than LSH.**

  On clustered L2 data, PQ with `m=32, nbits=8` achieved **0.707 recall with a 3.97 MB index**, while LSH with 256 bits achieved **0.046 recall with a 3.97 MB index**.

* **PQ required much longer build times than LSH.**

  For the same clustered L2 comparison, PQ took **2.85s** to build while LSH took only **0.087s**. At the larger configurations, PQ with `m=32, nbits=10` took **11.60s**, while LSH with 4096 bits took **3.98s**. This reflects the difference between a data-dependent method, which must run k-means over the database, and a data-independent one, which only generates random hyperplanes.

* **LSH's build-time advantage does not survive tuning.**

  LSH builds in **0.087s** at 256 bits, but returns only **0.046** recall there. Reaching **0.457** requires 4096 bits and **3.98s**, by which point PQ at `nbits=8` has better recall (**0.707**), a **16×** smaller index (3.97 MB against 63.50 MB) and a faster build (**2.85s**). The advantage exists only in the range where the accuracy is not usable.

* **HNSW gave the best recall and the lowest latency on clustered data.**

  At `M=32, efSearch=64` HNSW reached **0.986** recall at **0.0032s**, against **0.839** at **0.300s** for PQ (`m=32, nbits=10`) and **0.457** at **0.167s** for LSH (4096 bits).

* **At a matched storage budget the gap between HNSW and LSH is very large.**

  On clustered data, HNSW at `M=32` occupies **64.48 MB** and LSH at 4096 bits occupies **63.50 MB**, with build times of **4.07s** and **3.98s** respectively. For effectively the same memory and build cost, HNSW returned **0.986** recall against **0.457**, and was about **52×** faster to search.

* **The three methods trade different resources.**

  PQ compresses the vectors, so it is by far the smallest (**0.86-5.97 MB**) but scans every code and is the slowest to search. HNSW stores vectors uncompressed plus a graph, so it is the largest (**41.09-112.17 MB**) but by far the fastest. LSH compresses to binary codes but, on these datasets, obtained neither the smallest useful index nor competitive recall.

* **Dataset structure affects the three methods in opposite directions.**

  Moving from random to clustered data changed recall as follows, at each method's `M=32`/`m=32` configuration: PQ fell slightly (**0.865 → 0.839**), LSH fell sharply (**0.727 → 0.457** at 4096 bits), and HNSW rose sharply (**0.776 → 0.998** at `efSearch=128`). Clustering introduces correlations that sequential slicing cannot model for PQ, compresses the angular separation that LSH depends on, and provides the traversal gradient that HNSW needs. No single dataset property predicts performance across all three methods.

* **Parameter resolution scales differently across the methods.**

  PQ gains 2^`nbits` centroids per subspace for each additional bit, so its accuracy rises quickly with resolution: it reached **0.839** using 32 × 10 = **320 bits** per vector. LSH gains one hyperplane per bit, a linear return, and needed **4096 bits** per vector to reach only **0.512**. HNSW does not compress at all, and instead exposes `efSearch` as a search-time dial, tuning accuracy without changing the stored representation.

---

## Open questions / to revisit
-

---