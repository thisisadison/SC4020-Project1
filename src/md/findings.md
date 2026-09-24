# Findings — d = 384

All datasets d=384. Synthetic: 100,000 vectors, 100 queries. FinanceBench: 104,169 passages, 150 questions.
Metric: every vector is L2-normalised before indexing, and the ground truth is the exact top-10 under L2 distance (`IndexFlatL2`). PQ and HNSW search with L2; LSH hashes angles, which on unit vectors gives the same ranking (||x-y||^2 = 2 - 2cos).
Speedup = Flat latency / method latency, measured in the same run, so database size and query count cancel out.

---

## LSH

### `run_lsh_random()` — random data, L2 ground truth, k=10

**Baseline (Flat):** build 0.023s · latency 0.4509s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0210 | 0.21 | 0.0080 | 56.7× | 3.43 |
| 512 | 0.0980 | 0.14 | 0.0080 | 56.7× | 6.85 |
| 1024 | 0.1930 | 0.42 | 0.0216 | 20.9× | 13.71 |
| 2048 | 0.3210 | 1.22 | 0.0455 | 9.9× | 27.41 |
| 4096 | 0.4720 | 5.88 | 0.0660 | 6.8× | 54.83 |

### `run_lsh_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat):** build 0.022s · latency 0.3532s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0090 | 0.08 | 0.0078 | 45.2× | 3.43 |
| 512 | 0.0240 | 0.15 | 0.0084 | 42.1× | 6.85 |
| 1024 | 0.0430 | 0.29 | 0.0152 | 23.2× | 13.71 |
| 2048 | 0.0980 | 1.09 | 0.0248 | 14.2× | 27.41 |
| 4096 | 0.1660 | 5.42 | 0.1158 | 3.1× | 54.83 |

### `run_lsh_finance()` — FinanceBench, L2 ground truth, k=10

**Baseline (Flat):** build 0.079s · latency 0.6237s · size 152.59 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.3140 | 0.32 | 0.0347 | 18.0× | 3.55 |
| 512 | 0.4873 | 0.41 | 0.0488 | 12.8× | 7.11 |
| 1024 | 0.6133 | 0.93 | 0.0585 | 10.7× | 14.22 |
| 2048 | 0.6993 | 2.21 | 0.1426 | 4.4× | 28.43 |
| 4096 | 0.7913 | 9.76 | 0.2019 | 3.1× | 56.86 |

---

## PQ

### `run_pq_random()` — random data, L2 ground truth, k=10

**Baseline (Flat):** build 0.022s · latency 0.3506s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0650 | 2.32 | 0.1610 | 2.2× | 2.38 |
| 8 | 0.1060 | 30.48 | 0.0331 | 10.6× | 3.43 |
| 10 | 0.1750 | 203.09 | 0.2287 | 1.5× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0210 | 62.13 | 0.0480 | 7.3× | 2.45 |
| 16 | 0.0500 | 144.99 | 0.0845 | 4.1× | 3.41 |
| 32 | 0.1750 | 203.09 | 0.2287 | 1.5× | 5.31 |
| 64 | 0.3470 | 39.84 | 0.4634 | 0.8× | 9.13 |
| 192 | 0.8530 | 51.89 | 1.4797 | 0.2× | 24.39 |

### `run_pq_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat):** build 0.021s · latency 0.3686s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0560 | 2.29 | 0.1650 | 2.2× | 2.38 |
| 8 | 0.0930 | 30.53 | 0.0418 | 8.8× | 3.43 |
| 10 | 0.1610 | 213.99 | 0.2540 | 1.5× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0460 | 64.34 | 0.0426 | 8.7× | 2.45 |
| 16 | 0.0790 | 170.27 | 0.0852 | 4.3× | 3.41 |
| 32 | 0.1610 | 213.99 | 0.2540 | 1.5× | 5.31 |
| 64 | 0.3650 | 43.10 | 0.4861 | 0.8× | 9.13 |
| 192 | 0.8510 | 56.39 | 1.5708 | 0.2× | 24.39 |

### `run_pq_finance()` — FinanceBench, L2 ground truth, k=10

**Baseline (Flat):** build 0.028s · latency 0.6298s · size 152.59 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.3847 | 3.13 | 0.3704 | 1.7× | 2.48 |
| 8 | 0.4893 | 32.60 | 0.0728 | 8.7× | 3.55 |
| 10 | 0.5533 | 221.14 | 0.4083 | 1.5× | 5.47 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.2360 | 67.81 | 0.0995 | 6.3× | 2.49 |
| 16 | 0.3840 | 166.06 | 0.2522 | 2.5× | 3.49 |
| 32 | 0.5533 | 221.14 | 0.4083 | 1.5× | 5.47 |
| 64 | 0.7553 | 44.81 | 0.7395 | 0.9× | 9.45 |
| 192 | 0.9567 | 54.37 | 2.3103 | 0.3× | 25.34 |

---

## HNSW

### `run_hnsw_random()` — random data, L2 ground truth, k=10

**Baseline (Flat):** build 0.020s · latency 0.3736s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.0450 | 27.34 | 0.0093 | 40.1× | 172.44 |
| 32 | 0.0810 | 27.34 † | 0.0271 | 13.8× | 172.44 |
| 64 | 0.1330 | 27.34 † | 0.0307 | 12.2× | 172.44 |
| 128 | 0.2200 | 27.34 † | 0.0650 | 5.7× | 172.44 |
| 256 | 0.3770 | 27.34 † | 0.1226 | 3.0× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0380 | 8.25 | 0.0082 | 45.3× | 154.17 |
| 16 | 0.0790 | 14.16 | 0.0189 | 19.8× | 160.25 |
| 32 | 0.1330 | 27.34 † | 0.0307 | 12.2× | 172.44 |
| 64 | 0.1810 | 36.08 | 0.0520 | 7.2× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_clustered()` — clustered+correlated data, L2 ground truth, k=10

**Baseline (Flat):** build 0.041s · latency 0.5839s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.5240 | 23.31 | 0.0076 | 76.5× | 172.44 |
| 32 | 0.6780 | 23.31 † | 0.0164 | 35.6× | 172.44 |
| 64 | 0.8160 | 23.31 † | 0.0185 | 31.5× | 172.44 |
| 128 | 0.9240 | 23.31 † | 0.0626 | 9.3× | 172.44 |
| 256 | 0.9820 | 23.31 † | 0.1301 | 4.5× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.4540 | 7.04 | 0.0051 | 113.8× | 154.17 |
| 16 | 0.6510 | 8.73 | 0.0110 | 53.1× | 160.25 |
| 32 | 0.8160 | 23.31 † | 0.0185 | 31.5× | 172.44 |
| 64 | 0.8850 | 28.82 | 0.0293 | 19.9× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_finance()` — FinanceBench, L2 ground truth, k=10

**Baseline (Flat):** build 0.042s · latency 0.7461s · size 152.59 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.7807 | 10.79 | 0.0073 | 101.9× | 179.64 |
| 32 | 0.8800 | 10.79 † | 0.0064 | 115.9× | 179.64 |
| 64 | 0.9307 | 10.79 † | 0.0108 | 68.9× | 179.64 |
| 128 | 0.9847 | 10.79 † | 0.0330 | 22.6× | 179.64 |
| 256 | 0.9893 | 10.79 † | 0.0519 | 14.4× | 179.64 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.8140 | 3.81 | 0.0071 | 105.0× | 160.60 |
| 16 | 0.9013 | 4.67 | 0.0080 | 93.5× | 166.93 |
| 32 | 0.9307 | 10.79 † | 0.0108 | 68.9× | 179.64 |
| 64 | 0.9647 | 10.21 | 0.0193 | 38.6× | 205.06 |

† shared M=32 index — efSearch changes at search time, no rebuild.

---

# Observations and conclusions — d = 384

Format: **claim**: evidence from the tables → *why* (theory).
EDA numbers are computed on the normalised random and clustered vectors. FinanceBench EDA is still pending.

## 0. Setup and validity checks

- **One ground truth for every method.** Every vector is L2-normalised, and the ground truth is the exact top-10 under L2 distance.
  - PQ and HNSW search with L2.
  - LSH hashes angles, which on unit vectors gives the same ranking (‖x−y‖² = 2 − 2cos θ).
  - So every method runs in its native form against the same neighbours.
- **Every index size matches its formula** (codes + codebook for PQ, codes + rotation for LSH, vectors + links for HNSW): 0 mismatches across 60 rows.
- **Recall and index size are deterministic, except for HNSW recall; latency is noisy.**
  - Flat latency ranges from 0.35s to 0.75s between runs on data of the same size. So every speed comparison uses speedup against the Flat baseline from the same run.
  - LSH and PQ recall reproduce exactly between runs.
  - HNSW recall moves by up to about ±0.03 between runs (random, efSearch=256: 0.403 → 0.377). *FAISS inserts nodes in parallel, so the graph differs slightly each time.*

## 1. LSH

- **Recall rises steadily with every doubling of bits:**
  - random: 0.021 → 0.098 → 0.193 → 0.321 → 0.472;
  - FinanceBench: 0.314 → 0.487 → 0.613 → 0.699 → 0.791.
  - *Each bit is one random hyperplane, and two vectors at angle θ share a bit with probability 1 − θ/π. Hamming distance is therefore a noisy estimate of the angle, and the noise shrinks as bits are added.*
- **Cost grows with bits.** Size doubles with each doubling (3.43 → 54.83 MB), and latency grows about 6–15× from 256 to 4096 bits (clustered 0.0078 → 0.116s).
- **Cheapest build of all three methods:** 0.08–9.8s, with no training. *The hyperplanes are random and don't depend on the data.*
- **Never larger than Flat at d=384.** 4096 bits is 2.7× smaller than Flat (54.8 vs 146.5 MB). *A raw vector takes 384 × 32 = 12,288 bits.*
- **Weak on synthetic data:** best 0.472 on random and 0.166 on clustered, against 0.791 on FinanceBench. §5 gives the reason.

## 2. PQ

- **Recall is governed by dimensions per subspace (d/m).** With nbits=10, d/m = 48 → 24 → 12 → 6 → 2 gives:
  - random: 0.021 → 0.050 → 0.175 → 0.347 → 0.853;
  - FinanceBench: 0.236 → 0.384 → 0.553 → 0.755 → 0.957.
  - *For a fixed number of centroids, quantisation distortion grows with subspace dimension (∝ k^(−2/dsub)). 1,024 centroids cover 2 dimensions densely but 48 dimensions very sparsely.*
- **`nbits` matters much less than `m`.** At m=32, going from 64 to 1,024 centroids (nbits 6 → 10) moves random recall only from 0.065 to 0.175. Going from m=32 to m=192 at the same nbits moves it from 0.175 to 0.853.
- **Largest compression of any method:** 6–62× smaller than Flat (2.4–25.3 MB). *Each vector is stored as m small integers instead of 384 floats.*
- **`nbits=8` is the fastest PQ setting on every dataset:** 8.7–10.6× faster than Flat, against about 2× at 6 bits and 1.5× at 10 bits. *8-bit codes are exactly one byte, so lookups are byte-aligned. 6- and 10-bit codes have to be unpacked from shared bytes.*
- **At high recall PQ is slower than exact search.** m=192 runs at 0.2–0.3× Flat's speed on every dataset. *Plain PQ still scans every code (O(n)), doing m table lookups per vector, while Flat uses vectorised matrix multiplication. This is what motivates IVF (§7).*
- **Most expensive build, growing steeply with nbits.** At m=32 it takes 2.3s → 30.5s → 203.1s for nbits 6 → 8 → 10. *k-means cost scales with the number of centroids, 2^nbits.*
- **Build time does not increase steadily with m.** Random takes 62 / 145 / 203 / 40 / 52s for m = 8 / 16 / 32 / 64 / 192, and the same pattern appears on all three datasets. This is reported as observed; it is likely an implementation effect.
- **PQ needs enough training data.** FAISS warns below about 39 training points per centroid (about 40,000 for nbits=10). The first FinanceBench corpus (1,198 cited passages) could only support nbits ≤ 4, which is why full filings were used. *LSH and HNSW don't have this dependence on data.*

## 3. HNSW

- **`efSearch` tunes accuracy at search time, with no rebuild.** One FinanceBench index spans 0.781 → 0.989 recall as efSearch goes 16 → 256. *efSearch is the size of the candidate list kept during the greedy graph walk. Neither PQ nor LSH can change accuracy without a rebuild.*
- **Diminishing returns in efSearch.** Going 128 → 256 gains only +0.005 on FinanceBench (for 1.6× the latency) and +0.058 on clustered.
- **`M` trades memory for recall.** At efSearch=64:
  - FinanceBench: M = 8 / 16 / 32 / 64 gives 0.814 / 0.901 / 0.931 / 0.965;
  - clustered: 0.454 / 0.651 / 0.816 / 0.885.
- **Always larger than Flat**, by 5% (M=8) to 34% (M=64).
  - The overhead is 81 / 144 / 272 / 528 bytes per vector, roughly 2·M·4 bytes.
  - *HNSW stores the raw vectors plus M neighbour IDs per node (2M at the base layer). That overhead doesn't depend on d.*
- **Fastest method at high recall:**
  - FinanceBench: 0.985 recall at 23× faster than Flat, and 0.989 at 14×;
  - clustered: 0.924 at 9×, and 0.982 at 4.5×.
  - *The greedy walk visits only a small fraction of the database.*
- **Worst on random data:** best 0.377, and still climbing (+0.157 from efSearch 128 → 256), so not yet saturated. *See §5: random data gives the walk nothing to follow.*
- **Builds faster on easier data.** At M=32: FinanceBench 10.8s, clustered 23.3s, random 27.3s. *Building the graph is itself a series of searches, so data that is easy to search is also quick to index.*

## 4. Method comparison

- **Each method wins a different resource:**

  | | memory | search speed at high recall | build / training |
  |---|---|---|---|
  | LSH | medium (3.4–57 MB) | fast, but low recall | **cheapest (≤ 10s, no training)** |
  | PQ | **smallest (2.4–25 MB)** | slower than Flat | most expensive (up to 221s) |
  | HNSW | largest (154–205 MB, more than Flat) | **fastest (0.99 recall at 14× Flat)** | moderate (4–36s) |

- **At equal memory, LSH is faster and PQ is more accurate.** Both use 32 bytes per vector (LSH 256 bits vs PQ m=32, nbits=8). On FinanceBench: recall 0.314 vs 0.489, latency 0.035s vs 0.073s. *Hamming distance takes a few XOR + popcount instructions per vector; PQ does m table lookups but reconstructs a much better distance estimate.*
- **Only HNSW reaches high recall while staying faster than Flat.**
  - HNSW reaches ≥ 0.95 at 14–23× Flat's speed on FinanceBench.
  - PQ reaches 0.957 only at m=192, running at 0.3× Flat's speed.
  - LSH never gets there (best 0.791).
- **The order LSH → PQ → HNSW follows each method's weakness.**
  - PQ improves on LSH per bit because its codes are learned from the data.
  - HNSW avoids compression altogether and instead avoids comparing the query against most of the database.

## 5. Effect of data

- **Best recall per dataset (largest config tested):**

  | | random | clustered | FinanceBench |
  |---|---|---|---|
  | LSH (4096 bits) | 0.472 | 0.166 | **0.791** |
  | PQ (m=192, nbits=10) | 0.853 | 0.851 | **0.957** |
  | HNSW (M=32, efSearch=256) | 0.377 | 0.982 | **0.989** |

- **All three methods do best on real data**, even though every dataset has 384 dimensions. *Text embeddings lie near a much lower-dimensional structure (companies, statement types, topics). Random data genuinely uses all 384 dimensions: 323 of them are needed to keep 90% of the variance (eda4).*
- **HNSW depends on global contrast.**
  - On random data the average point is only 1.13× as far away as the nearest one; on clustered data it is 6.2×.
  - HNSW recall follows: 0.377 vs 0.982.
  - *Each greedy hop moves to the neighbour closest to the query. When every point is almost equally far away, hops make no progress.*
- **LSH depends on angle gaps, and clusters shrink them.**
  - Inside a cluster, the 10th and 1000th neighbours differ by only 1.0°; on random data they differ by 4.1°.
  - *Each bit agrees with probability 1 − θ/π, so the clustered gap carries about 4× less information per bit.* Taking the noise of each bit into account, clustered data needs about 4–5× more bits to do as well.
  - The data bears this out: clustered at 4096 bits (0.166) sits between random at 512 bits (0.098) and at 1024 bits (0.193).
  - So data that is easy for a graph to search can still be hard for LSH.
- **PQ is indifferent to cluster structure:** random 0.853, clustered 0.851 at m=192. *PQ has to rank the neighbours inside the query's own cluster, and that local problem is about as hard in both datasets (distance ratio between the 10th and 100th neighbour: 0.981 vs 0.965).*
- **On random data LSH currently beats HNSW (0.472 vs 0.377).** This is tentative, because HNSW had not saturated at efSearch=256 and M=64 was only tested at efSearch=64.

## 6. Conclusions: which method when

| situation | choose | because |
|---|---|---|
| memory is the binding constraint | **PQ** (nbits=8 for speed) | 6–62× compression |
| latency-critical, high recall needed | **HNSW** | ≥ 0.95 recall at 14–23× Flat's speed on real data |
| data changes constantly / no training data / cheapest comparisons | **LSH** | no training, builds in seconds, Hamming distance |
| data is isotropic and high-dimensional | none performs well | curse of dimensionality: reconsider the representation |

## 7. Improvements, limitations, open items

- **Improvements (discussed):**
  - **IVF:** partition the database into cells and probe only a few, making PQ's scan sub-linear. This addresses PQ being slower than Flat.
  - **OPQ:** learn a rotation before quantising, so correlated dimensions share a subspace.
  - **HNSW + PQ:** store PQ codes in the graph to cut HNSW's memory.
- **Limitations:**
  - CPU-only laptop timings, with run-to-run noise.
  - One embedding model, and 150 FinanceBench queries.
  - The synthetic data is Gaussian.
  - Normalisation discards vector length, so the study measures direction-based similarity. Conclusions may not transfer to tasks where length carries meaning.
  - FAISS `IndexLSH` scans every code (no hash-bucket lookup), so LSH is measured as compact codes rather than bucketed retrieval.
  - Parameters were swept one at a time around M=32 / efSearch=64 and m=32 / nbits=10, not as a full grid.
- **Open items:**
  - A full parameter grid, especially HNSW M × efSearch on random data, to settle LSH vs HNSW.
  - FinanceBench EDA.
  - Saving chunk texts for the FinanceBench success and failure cases.
