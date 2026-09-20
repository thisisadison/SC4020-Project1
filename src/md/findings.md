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

## Open questions / to revisit
-

---