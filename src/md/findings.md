# Findings — d = 384

All datasets d=384. Synthetic: 100,000 vectors, 100 queries. FinanceBench: 104,169 passages, 150 questions.
Speedup = Flat latency / method latency, measured in the same run, so database size and query count cancel out.

---

## LSH

### `run_lsh_l2_random()` — random data, L2 ground truth, k=10

**Baseline (Flat):** build 0.033s · latency 0.3953s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0210 | 0.11 | 0.0077 | 51.4× | 3.43 |
| 512 | 0.0980 | 0.17 | 0.0142 | 27.9× | 6.85 |
| 1024 | 0.1930 | 0.45 | 0.0151 | 26.2× | 13.71 |
| 2048 | 0.3210 | 1.28 | 0.0329 | 12.0× | 27.41 |
| 4096 | 0.4720 | 7.69 | 0.1248 | 3.2× | 54.83 |

### `run_lsh_cosine_random()` — random data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.055s · latency 0.3557s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0210 | 0.18 | 0.0142 | 25.0× | 3.43 |
| 512 | 0.0980 | 0.20 | 0.0180 | 19.8× | 6.85 |
| 1024 | 0.1930 | 0.40 | 0.0181 | 19.7× | 13.71 |
| 2048 | 0.3210 | 1.26 | 0.0310 | 11.5× | 27.41 |
| 4096 | 0.4720 | 6.55 | 0.0828 | 4.3× | 54.83 |

### `run_lsh_l2_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat):** build 0.032s · latency 0.4296s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0080 | 0.08 | 0.0080 | 53.6× | 3.43 |
| 512 | 0.0170 | 0.16 | 0.0094 | 45.8× | 6.85 |
| 1024 | 0.0360 | 0.33 | 0.0167 | 25.7× | 13.71 |
| 2048 | 0.1000 | 1.77 | 0.1186 | 3.6× | 27.41 |
| 4096 | 0.1540 | 11.40 | 0.1802 | 2.4× | 54.83 |

### `run_lsh_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.098s · latency 0.4589s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0090 | 0.17 | 0.0130 | 35.4× | 3.43 |
| 512 | 0.0240 | 0.55 | 0.0145 | 31.6× | 6.85 |
| 1024 | 0.0430 | 1.00 | 0.0191 | 24.1× | 13.71 |
| 2048 | 0.0980 | 1.71 | 0.0627 | 7.3× | 27.41 |
| 4096 | 0.1660 | 8.40 | 0.1356 | 3.4× | 54.83 |

### `run_lsh_cosine_finance()` — FinanceBench, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.103s · latency 0.5597s · size 152.59 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.3140 | 0.10 | 0.0120 | 46.8× | 3.55 |
| 512 | 0.4873 | 0.15 | 0.0176 | 31.9× | 7.11 |
| 1024 | 0.6133 | 0.45 | 0.0499 | 11.2× | 14.22 |
| 2048 | 0.6993 | 1.00 | 0.0936 | 6.0× | 28.43 |
| 4096 | 0.7913 | 5.87 | 0.1904 | 2.9× | 56.86 |

---

## PQ

### `run_pq_l2_random()` — random data, L2 ground truth, k=10

**Baseline (Flat):** build 0.043s · latency 0.5076s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0660 | 2.39 | 0.1804 | 2.8× | 2.38 |
| 8 | 0.1040 | 32.35 | 0.0350 | 14.5× | 3.43 |
| 10 | 0.1780 | 204.26 | 0.2688 | 1.9× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0210 | 62.62 | 0.0570 | 8.9× | 2.45 |
| 16 | 0.0490 | 146.89 | 0.1307 | 3.9× | 3.41 |
| 32 | 0.1780 | 204.26 | 0.2688 | 1.9× | 5.31 |
| 64 | 0.3490 | 39.73 | 0.4643 | 1.1× | 9.13 |
| 192 | 0.8530 | 52.16 | 1.5162 | 0.3× | 24.39 |

### `run_pq_cosine_random()` — random data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.062s · latency 0.3741s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0620 | 2.09 | 0.1743 | 2.1× | 2.38 |
| 8 | 0.1050 | 30.47 | 0.0378 | 9.9× | 3.43 |
| 10 | 0.1970 | 202.65 | 0.2312 | 1.6× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0210 | 60.96 | 0.0417 | 9.0× | 2.45 |
| 16 | 0.0550 | 142.35 | 0.1319 | 2.8× | 3.41 |
| 32 | 0.1970 | 202.65 | 0.2312 | 1.6× | 5.31 |
| 64 | 0.4520 | 38.75 | 0.5494 | 0.7× | 9.13 |
| 192 | 0.9010 | 57.52 | 1.6617 | 0.2× | 24.39 |

### `run_pq_l2_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat):** build 0.039s · latency 0.4420s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0580 | 2.05 | 0.2116 | 2.1× | 2.38 |
| 8 | 0.1050 | 30.85 | 0.0432 | 10.2× | 3.43 |
| 10 | 0.1490 | 203.01 | 0.2515 | 1.8× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0390 | 61.30 | 0.0419 | 10.6× | 2.45 |
| 16 | 0.0940 | 144.08 | 0.1266 | 3.5× | 3.41 |
| 32 | 0.1490 | 203.01 | 0.2515 | 1.8× | 5.31 |
| 64 | 0.3740 | 38.91 | 0.4690 | 0.9× | 9.13 |
| 192 | 0.8530 | 50.72 | 1.4372 | 0.3× | 24.39 |

### `run_pq_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.062s · latency 0.3576s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0020 | 2.28 | 0.1659 | 2.2× | 2.38 |
| 8 | 0.0070 | 30.62 | 0.0326 | 11.0× | 3.43 |
| 10 | 0.0110 | 203.16 | 0.2275 | 1.6× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0060 | 61.26 | 0.0641 | 5.6× | 2.45 |
| 16 | 0.0060 | 147.69 | 0.1127 | 3.2× | 3.41 |
| 32 | 0.0110 | 203.16 | 0.2275 | 1.6× | 5.31 |
| 64 | 0.0230 | 41.08 | 0.4635 | 0.8× | 9.13 |
| 192 | 0.4910 | 53.89 | 1.6540 | 0.2× | 24.39 |

### `run_pq_cosine_finance()` — FinanceBench, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.418s · latency 0.6377s · size 152.59 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.3433 | 2.88 | 0.3734 | 1.7× | 2.48 |
| 8 | 0.4573 | 31.97 | 0.0637 | 10.0× | 3.55 |
| 10 | 0.5187 | 213.94 | 0.3663 | 1.7× | 5.47 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.2207 | 65.22 | 0.0642 | 9.9× | 2.49 |
| 16 | 0.3313 | 148.66 | 0.2317 | 2.8× | 3.49 |
| 32 | 0.5187 | 213.94 | 0.3663 | 1.7× | 5.47 |
| 64 | 0.7267 | 42.15 | 0.7053 | 0.9× | 9.45 |
| 192 | 0.9553 | 54.26 | 2.2510 | 0.3× | 25.34 |

---

## HNSW

### `run_hnsw_l2_random()` — random data, L2 ground truth, k=10

**Baseline (Flat):** build 0.033s · latency 0.4196s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.0380 | 32.65 | 0.0133 | 31.6× | 172.44 |
| 32 | 0.0810 | 32.65 † | 0.0181 | 23.2× | 172.44 |
| 64 | 0.1320 | 32.65 † | 0.0295 | 14.2× | 172.44 |
| 128 | 0.2400 | 32.65 † | 0.0560 | 7.5× | 172.44 |
| 256 | 0.3960 | 32.65 † | 0.1708 | 2.5× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0280 | 10.85 | 0.0101 | 41.6× | 154.17 |
| 16 | 0.0680 | 19.35 | 0.0180 | 23.3× | 160.25 |
| 32 | 0.1320 | 32.65 † | 0.0295 | 14.2× | 172.44 |
| 64 | 0.1890 | 34.72 | 0.0603 | 7.0× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_cosine_random()` — random data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.076s · latency 0.4139s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.0420 | 24.72 | 0.0148 | 27.9× | 172.44 |
| 32 | 0.0760 | 24.72 † | 0.0238 | 17.4× | 172.44 |
| 64 | 0.1450 | 24.72 † | 0.0290 | 14.3× | 172.44 |
| 128 | 0.2410 | 24.72 † | 0.0615 | 6.7× | 172.44 |
| 256 | 0.3910 | 24.72 † | 0.1629 | 2.5× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0300 | 7.72 | 0.0079 | 52.2× | 154.17 |
| 16 | 0.0700 | 13.41 | 0.0175 | 23.7× | 160.25 |
| 32 | 0.1450 | 24.72 † | 0.0290 | 14.3× | 172.44 |
| 64 | 0.2140 | 30.94 | 0.0753 | 5.5× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_l2_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat):** build 0.094s · latency 0.4457s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.5290 | 19.76 | 0.0095 | 46.8× | 172.44 |
| 32 | 0.6640 | 19.76 † | 0.0109 | 40.9× | 172.44 |
| 64 | 0.8150 | 19.76 † | 0.0183 | 24.3× | 172.44 |
| 128 | 0.9290 | 19.76 † | 0.0326 | 13.7× | 172.44 |
| 256 | 0.9770 | 19.76 † | 0.0561 | 7.9× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.4880 | 4.74 | 0.0059 | 75.1× | 154.17 |
| 16 | 0.6570 | 8.08 | 0.0110 | 40.6× | 160.25 |
| 32 | 0.8150 | 19.76 † | 0.0183 | 24.3× | 172.44 |
| 64 | 0.8990 | 25.27 | 0.0313 | 14.2× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.142s · latency 0.4124s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.5090 | 20.13 | 0.0216 | 19.1× | 172.44 |
| 32 | 0.6730 | 20.13 † | 0.0160 | 25.7× | 172.44 |
| 64 | 0.8200 | 20.13 † | 0.0225 | 18.3× | 172.44 |
| 128 | 0.9200 | 20.13 † | 0.0361 | 11.4× | 172.44 |
| 256 | 0.9820 | 20.13 † | 0.0852 | 4.8× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.4790 | 5.15 | 0.0101 | 40.6× | 154.17 |
| 16 | 0.6480 | 8.71 | 0.0149 | 27.6× | 160.25 |
| 32 | 0.8200 | 20.13 † | 0.0225 | 18.3× | 172.44 |
| 64 | 0.8850 | 33.11 | 0.0716 | 5.8× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_cosine_finance()` — FinanceBench, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.050s · latency 0.6062s · size 152.59 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.7747 | 8.13 | 0.0071 | 85.0× | 179.64 |
| 32 | 0.8953 | 8.13 † | 0.0087 | 69.6× | 179.64 |
| 64 | 0.9313 | 8.13 † | 0.0109 | 55.5× | 179.64 |
| 128 | 0.9853 | 8.13 † | 0.0234 | 25.9× | 179.64 |
| 256 | 0.9887 | 8.13 † | 0.0406 | 14.9× | 179.64 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.8153 | 3.80 | 0.0079 | 76.5× | 160.60 |
| 16 | 0.8993 | 4.66 | 0.0064 | 95.4× | 166.93 |
| 32 | 0.9313 | 8.13 † | 0.0109 | 55.5× | 179.64 |
| 64 | 0.9673 | 10.49 | 0.0175 | 34.7× | 205.06 |

† shared M=32 index — efSearch changes at search time, no rebuild.

---

# Observations and conclusions — d = 384

Format: **claim** — evidence from the tables — *why* (theory).

## 0. Validity checks

- **Every index size matches its formula** (codes + codebook for PQ, codes + rotation for LSH, vectors + links for HNSW) — 0 mismatches across 100 rows. The data, dimension and indexes are wired correctly.
- **Recall and index size are deterministic; latency is not.** A few timings wobble between near-identical runs (e.g. HNSW clustered M=64: 0.031s under L2 vs 0.072s under cosine). All speed comparisons use speedup vs Flat from the same run, which also cancels the effect of database size and query count (FinanceBench Flat is slower, at 0.56–0.64s vs 0.36–0.51s, because it has 104k vectors and 150 queries).

## 1. LSH

- **Recall rises steadily with every doubling of bits** — random: 0.021 → 0.098 → 0.193 → 0.321 → 0.472; FinanceBench: 0.314 → 0.487 → 0.613 → 0.699 → 0.791. *Each bit is one random hyperplane, and two vectors at angle θ share a bit with probability 1 − θ/π. Hamming distance is therefore a noisy estimate of the angle whose variance falls as 1/nbits — more bits, sharper estimate.*
- **Cost grows linearly with bits** — size doubles per doubling (3.43 → 54.83 MB); latency grows 16× from 256 to 4096 bits (random: 0.0077 → 0.125s).
- **Cheapest build of all three methods** — 0.08–11s, with no training step. *Hyperplanes are drawn at random, independent of the data.*
- **Never exceeds Flat's size at d=384** — 4096 bits is 2.7× smaller than Flat (54.8 vs 146.5 MB). *Raw vectors cost 384 × 32 = 12,288 bits each, so compression ratios depend on d; at d=64 the same 4096-bit index was larger than Flat.*
- **Weak on high-dimensional synthetic data** — best 0.472 (random), 0.166 (clustered). *In 384-D, cosine between random vectors has spread ≈ 1/√384 ≈ 0.05, so neighbours are barely closer in angle than everything else and random hyperplanes rarely separate them.*
- **L2 and cosine give identical recall on random data** at all five bit counts. *The data is unit-norm, and a hash bit (the sign of x·r) is unchanged by scaling x — so both the ground truth and the hashes coincide.*

## 2. PQ

- **Recall is governed by dimensions per subspace (d/m)** — random L2 at nbits=10: d/m = 48 → 24 → 12 → 6 → 2 gives 0.021 → 0.049 → 0.178 → 0.349 → 0.853. *Quantisation distortion grows with subspace dimension for a fixed number of centroids (∝ k^(−2/dsub)); 1,024 centroids cover 2 dims densely but 48 dims very sparsely.*
- **`nbits` matters much less than `m`** — at m=32, going from 64 to 1,024 centroids (nbits 6 → 10) only moves random L2 recall 0.066 → 0.178. *With 12 dims per subspace, even 1,024 centroids are sparse; splitting into more subspaces helps far more than adding centroids.*
- **Largest compression of any method** — 6–62× smaller than Flat (2.4–25.3 MB vs 146–153 MB). *Each vector is stored as m small integers instead of 384 floats.*
- **`nbits=8` is the fastest PQ setting on every dataset** — ~10–15× faster than Flat (random 14.5×, clustered 10.2×, FinanceBench 10.0×), against ~2× at 6 and 10 bits. *8-bit codes are exactly one byte, so lookups are byte-aligned; 6- and 10-bit codes must be unpacked from shared bytes.*
- **At high recall PQ is slower than exact search** — m=192 runs at 0.2–0.3× Flat's speed on all datasets. *Plain PQ still scans every code (O(n)), with m random-access table lookups per vector, while Flat uses vectorised matrix multiplication. This motivates IVF (§8).*
- **Build time is the largest of the three methods and grows steeply with nbits** — m=32: 2s → 31s → 204s for nbits 6 → 8 → 10. *k-means cost scales with the number of centroids, 2^nbits.*
- **Build time is not monotonic in m** — 62 / 147 / 204 / 40 / 52s for m = 8 / 16 / 32 / 64 / 192, reproduced within a few seconds under both metrics. Reported as observed; likely an implementation effect, not explained by theory.
- **PQ needs enough training data** — FAISS warns below ~39 training points per centroid (≈ 40,000 for nbits=10). The first FinanceBench corpus (1,198 cited passages) could only support nbits ≤ 4, which is why full filings were used. *A cost of data dependence that LSH and HNSW do not share.*

## 3. HNSW

- **`efSearch` tunes accuracy at search time, with no rebuild** — one FinanceBench index spans 0.775 → 0.989 recall as efSearch goes 16 → 256. *efSearch is the size of the candidate list kept during the greedy graph walk; neither PQ nor LSH can change accuracy without rebuilding.*
- **Diminishing returns in efSearch** — FinanceBench 128 → 256 gains only +0.003 recall for 1.7× the latency; clustered gains +0.048.
- **`M` trades memory for recall** — FinanceBench at efSearch=64: M = 8 / 16 / 32 / 64 gives 0.815 / 0.899 / 0.931 / 0.967.
- **Always larger than Flat** — 5% (M=8) to 34% (M=64) over Flat. The overhead is 81 / 144 / 272 / 528 bytes per vector, ≈ 2·M·4 bytes. *HNSW stores the raw vectors plus M neighbour IDs per node (2M at the base layer); the overhead is independent of d, identical to the d=64 runs.*
- **Fastest method at high recall** — FinanceBench 0.985 recall at 25.9× faster than Flat; clustered 0.977 at 7.9×. *The greedy walk visits only a small fraction of the database.*
- **Worst on random data** — best 0.396, and still climbing (+0.156 from efSearch 128 → 256), so not yet saturated. *Isotropic 384-D Gaussian data has no local structure: all points are nearly equidistant, so the greedy walk has no gradient to follow.*
- **Builds faster on easier data** — M=32: FinanceBench 8.1s, clustered 19.8s, random 32.6s. *Graph construction is itself a series of searches, so data that is easy to search is also quick to index.*

## 4. Method comparison

- **Each method wins a different resource:**

  | | memory | search speed at high recall | build / training |
  |---|---|---|---|
  | LSH | medium (3.4–57 MB) | fast, but low recall | **cheapest (≤ 11s, no training)** |
  | PQ | **smallest (2.4–25 MB)** | slower than Flat | most expensive (up to 214s) |
  | HNSW | largest (154–205 MB, > Flat) | **fastest (up to 95× Flat)** | moderate (4–35s) |

- **At equal memory, LSH is faster and PQ is more accurate** — both at 32 bytes per vector (LSH 256 bits vs PQ m=32, nbits=8), FinanceBench: recall 0.314 vs 0.457, latency 0.012s vs 0.064s. *Hamming distance is a few XOR + popcount instructions per vector; PQ does m table lookups and additions but reconstructs a much better distance estimate.*
- **Only HNSW reaches high recall while staying faster than Flat** — ≥ 0.95 recall: HNSW yes (25.9× on FinanceBench); PQ reaches 0.955 only at m=192, 0.3× Flat's speed; LSH never reaches it (best 0.791).
- **PQ improves on LSH per bit because its codes are learned from the data**; HNSW sidesteps compression and instead avoids comparing against most of the database. The ordering LSH → PQ → HNSW follows both the history of the methods and the weakness each one addresses.

## 5. Effect of distance metric (L2 vs cosine)

- **On unit-norm data the two metrics are equivalent** — random: LSH identical at all 5 configs, HNSW within ±0.025. *For unit vectors ‖x−y‖² = 2 − 2·cos(x, y), so both rank neighbours identically.*
- **Inner-product PQ is at least as good as L2 PQ on unit-norm data** — random: ≥ in 6 of 7 configs, gains up to +0.103 (m=64) and +0.048 (m=192); the one loss is −0.004 at near-zero recall. *‖q−ŷ‖² = 1 + ‖ŷ‖² − 2q·ŷ: quantised codewords ŷ are not unit length, so the L2 score carries an error term that a cosine ground truth ignores; inner product drops it.*
- **On non-normalised data, normalising changes the task — and only PQ suffers** — clustered, cosine minus L2: LSH within ±0.012, HNSW within ±0.02, PQ lower at every config, by up to −0.36 (m=192: 0.853 → 0.491; m=64: 0.374 → 0.023).
- **Ablation: quantisation error is the cause of PQ's drop** — HNSW searches the same normalised vectors without compressing them and loses nothing, so the neighbour structure survives normalisation; what fails is PQ's compressed representation. *Normalising squeezes each cluster into a narrow cone, so the differences between true neighbours become smaller than PQ's quantisation error.*
- **Takeaway:** use cosine when vector magnitude is noise (text embeddings), and L2 when magnitude carries structure. The choice matters most for compressed methods.

## 6. Effect of data

- **Best recall per dataset (cosine, largest config tested):**

  | | random | clustered | FinanceBench |
  |---|---|---|---|
  | LSH (4096 bits) | 0.472 | 0.166 | **0.791** |
  | PQ (m=192, nbits=10) | 0.901 | 0.491 | **0.955** |
  | HNSW (M=32, efSearch=256) | 0.391 | 0.982 | **0.989** |

- **All three methods do best on real data** — despite having the same 384 dimensions. *Text embeddings lie near a much lower-dimensional structure (companies, statement types, topics), whereas random Gaussian data genuinely uses all 384 dimensions — the worst case for any ANN method.*
- **Data structure affects the methods in opposite directions** — going from random to clustered, HNSW rises (0.391 → 0.982) while LSH falls (0.472 → 0.166). *Clusters give the graph local neighbourhoods to navigate, but compress the angular separation that random hyperplanes rely on.*
- **Under L2, PQ is indifferent to cluster structure** — random and clustered both reach 0.853 at m=192. *A query's true neighbours lie inside its own cluster, which is itself a high-dimensional Gaussian, so the local quantisation problem is the same.*
- **On random data LSH currently beats HNSW (0.472 vs 0.391)** — tentative, since HNSW had not saturated at efSearch=256. *SimHash's collision probability depends only on angle, whereas graph navigation depends on local structure that random data lacks.*

## 7. Conclusions — which method when

| situation | choose | because |
|---|---|---|
| memory is the binding constraint | **PQ** (nbits=8 for speed) | 6–62× compression |
| latency-critical, high recall needed | **HNSW** | ≥ 0.95 recall at 5–26× Flat's speed on structured data |
| data changes constantly / no training data / cheapest possible comparisons | **LSH** | no training, sub-second builds, Hamming distance |
| data is isotropic and high-dimensional | none performs well | curse of dimensionality — reconsider the representation |

## 8. Improvements, limitations, open items

- **IVF** — partition the database into cells and probe only a few, making PQ's scan sub-linear; addresses PQ being slower than Flat.
- **OPQ** — learn a rotation before quantising, so correlated dimensions share a subspace; addresses PQ's sequential slicing.
- **HNSW + PQ** — store PQ codes in the graph to cut HNSW's memory.
- **Match the metric to the data** — normalise only when magnitude is noise.
- **Limitations:** CPU-only laptop timings with run-to-run noise; one embedding model; 150 FinanceBench queries; synthetic data is Gaussian; FAISS `IndexLSH` scans all codes (no hash-bucket lookup), so LSH is measured as compact codes, not bucketed retrieval.
- **Open item:** extend random-data HNSW to efSearch 512 / 1024 to confirm or overturn the LSH-vs-HNSW result.
