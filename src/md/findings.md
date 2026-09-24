# Findings — d = 384

All datasets d=384. Synthetic: 100,000 vectors, 100 queries. FinanceBench: 104,169 passages, 150 questions.
Metric: every vector is L2-normalised and the ground truth is cosine similarity (exact `IndexFlatIP` on the normalised vectors). On unit vectors cosine and L2 rank neighbours identically, so each method runs its native metric: angle for LSH, L2 for PQ, inner product for HNSW.
Speedup = Flat latency / method latency, measured in the same run, so database size and query count cancel out.

---

## LSH

### `run_lsh_cosine_random()` — random data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.145s · latency 0.4866s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0210 | 0.51 | 0.0251 | 19.4× | 3.43 |
| 512 | 0.0980 | 0.46 | 0.0125 | 38.9× | 6.85 |
| 1024 | 0.1930 | 0.59 | 0.0275 | 17.7× | 13.71 |
| 2048 | 0.3210 | 1.24 | 0.0549 | 8.9× | 27.41 |
| 4096 | 0.4720 | 5.81 | 0.1321 | 3.7× | 54.83 |

### `run_lsh_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.057s · latency 0.3617s · size 146.48 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.0090 | 0.09 | 0.0079 | 45.5× | 3.43 |
| 512 | 0.0240 | 0.13 | 0.0107 | 33.9× | 6.85 |
| 1024 | 0.0430 | 0.33 | 0.0195 | 18.5× | 13.71 |
| 2048 | 0.0980 | 1.24 | 0.0463 | 7.8× | 27.41 |
| 4096 | 0.1660 | 6.54 | 0.1072 | 3.4× | 54.83 |

### `run_lsh_cosine_finance()` — FinanceBench, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.285s · latency 0.5643s · size 152.59 MB

**nbits sweep**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 256 | 0.3140 | 0.13 | 0.0151 | 37.3× | 3.55 |
| 512 | 0.4873 | 0.12 | 0.0296 | 19.1× | 7.11 |
| 1024 | 0.6133 | 0.34 | 0.0505 | 11.2× | 14.22 |
| 2048 | 0.6993 | 0.80 | 0.1063 | 5.3× | 28.43 |
| 4096 | 0.7913 | 6.87 | 0.1794 | 3.1× | 56.86 |

---

## PQ

### `run_pq_cosine_random()` — random data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.048s · latency 0.3419s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0650 | 2.25 | 0.1674 | 2.0× | 2.38 |
| 8 | 0.1060 | 30.56 | 0.0351 | 9.7× | 3.43 |
| 10 | 0.1750 | 202.48 | 0.2236 | 1.5× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0210 | 64.73 | 0.0534 | 6.4× | 2.45 |
| 16 | 0.0500 | 148.75 | 0.1082 | 3.2× | 3.41 |
| 32 | 0.1750 | 202.48 | 0.2236 | 1.5× | 5.31 |
| 64 | 0.3470 | 38.06 | 0.4581 | 0.7× | 9.13 |
| 192 | 0.8530 | 50.81 | 1.4609 | 0.2× | 24.39 |

### `run_pq_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.069s · latency 0.3481s · size 146.48 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.0560 | 2.33 | 0.1619 | 2.2× | 2.38 |
| 8 | 0.0930 | 30.85 | 0.0345 | 10.1× | 3.43 |
| 10 | 0.1610 | 202.72 | 0.2324 | 1.5× | 5.31 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0460 | 58.65 | 0.0498 | 7.0× | 2.45 |
| 16 | 0.0790 | 154.63 | 0.1645 | 2.1× | 3.41 |
| 32 | 0.1610 | 202.72 | 0.2324 | 1.5× | 5.31 |
| 64 | 0.3650 | 43.34 | 0.5149 | 0.7× | 9.13 |
| 192 | 0.8510 | 59.29 | 1.5949 | 0.2× | 24.39 |

### `run_pq_cosine_finance()` — FinanceBench, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.048s · latency 0.5495s · size 152.59 MB

**nbits sweep — m=32**

| nbits | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 6 | 0.3847 | 2.12 | 0.2660 | 2.1× | 2.48 |
| 8 | 0.4893 | 31.04 | 0.0638 | 8.6× | 3.55 |
| 10 | 0.5533 | 216.60 | 0.4236 | 1.3× | 5.47 |

**m sweep — nbits=10**

| m | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.2360 | 66.30 | 0.1232 | 4.5× | 2.49 |
| 16 | 0.3840 | 155.19 | 0.2191 | 2.5× | 3.49 |
| 32 | 0.5533 | 216.60 | 0.4236 | 1.3× | 5.47 |
| 64 | 0.7553 | 55.52 | 0.9969 | 0.6× | 9.45 |
| 192 | 0.9567 | 75.68 | 3.2271 | 0.2× | 25.34 |

---

## HNSW

### `run_hnsw_cosine_random()` — random data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.054s · latency 0.3799s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.0420 | 25.68 | 0.0095 | 40.0× | 172.44 |
| 32 | 0.0770 | 25.68 † | 0.0144 | 26.4× | 172.44 |
| 64 | 0.1620 | 25.68 † | 0.0405 | 9.4× | 172.44 |
| 128 | 0.2590 | 25.68 † | 0.0596 | 6.4× | 172.44 |
| 256 | 0.4030 | 25.68 † | 0.1222 | 3.1× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.0300 | 7.91 | 0.0092 | 41.4× | 154.17 |
| 16 | 0.0660 | 14.17 | 0.0213 | 17.8× | 160.25 |
| 32 | 0.1620 | 25.68 † | 0.0405 | 9.4× | 172.44 |
| 64 | 0.1970 | 34.72 | 0.0589 | 6.5× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_cosine_clustered()` — clustered+correlated data, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.072s · latency 0.4324s · size 146.48 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.5150 | 19.35 | 0.0126 | 34.2× | 172.44 |
| 32 | 0.6710 | 19.35 † | 0.0120 | 36.1× | 172.44 |
| 64 | 0.8180 | 19.35 † | 0.0299 | 14.5× | 172.44 |
| 128 | 0.9130 | 19.35 † | 0.0333 | 13.0× | 172.44 |
| 256 | 0.9810 | 19.35 † | 0.1330 | 3.3× | 172.44 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.4550 | 5.55 | 0.0073 | 59.3× | 154.17 |
| 16 | 0.6340 | 8.59 | 0.0140 | 30.9× | 160.25 |
| 32 | 0.8180 | 19.35 † | 0.0299 | 14.5× | 172.44 |
| 64 | 0.8950 | 28.74 | 0.0315 | 13.7× | 196.85 |

† shared M=32 index — efSearch changes at search time, no rebuild.

### `run_hnsw_cosine_finance()` — FinanceBench, cosine ground truth, k=10

**Baseline (Cosine Flat):** build 0.144s · latency 0.7291s · size 152.59 MB

**efSearch sweep — M=32 (one index, reused)**

| efSearch | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 16 | 0.7880 | 10.92 | 0.0048 | 150.5× | 179.64 |
| 32 | 0.9060 | 10.92 † | 0.0064 | 114.4× | 179.64 |
| 64 | 0.9373 | 10.92 † | 0.0108 | 67.6× | 179.64 |
| 128 | 0.9807 | 10.92 † | 0.0385 | 19.0× | 179.64 |
| 256 | 0.9900 | 10.92 † | 0.0603 | 12.1× | 179.64 |

**M sweep — efSearch=64**

| M | recall@10 | build (s) | latency (s) | speedup vs Flat | size (MB) |
|---|---|---|---|---|---|
| 8 | 0.8080 | 5.24 | 0.0067 | 108.4× | 160.60 |
| 16 | 0.8867 | 6.16 | 0.0098 | 74.7× | 166.93 |
| 32 | 0.9373 | 10.92 † | 0.0108 | 67.6× | 179.64 |
| 64 | 0.9640 | 14.64 | 0.0281 | 26.0× | 205.06 |

† shared M=32 index — efSearch changes at search time, no rebuild.

---

# Observations and conclusions — d = 384, cosine

Format: **claim**: evidence from the tables → *why* (theory).
EDA numbers come from `eda.py` (random and clustered). FinanceBench EDA is still pending.

## 0. Setup and validity checks

- **One metric everywhere.** Every run uses the same cosine ground truth, so each recall number is comparable across methods and datasets.
  - Normalising is simply what cosine means, and it is applied to every dataset.
  - FinanceBench embeddings already come out unit length from all-MiniLM-L6-v2.
- **Every index size matches its formula** (codes + codebook for PQ, codes + rotation for LSH, vectors + links for HNSW): 0 mismatches across 60 rows.
- **Recall and index size are deterministic, except for HNSW recall; latency is noisy.**
  - Flat latency ranges from 0.34s to 0.73s between runs on identical data. So every speed comparison uses speedup against the Flat baseline from the same run.
  - LSH and PQ recall reproduce exactly between runs.
  - HNSW recall moves by about ±0.01 between runs (FinanceBench M=8: 0.815 → 0.808). *FAISS inserts nodes in parallel, so the graph differs slightly each time.*

## 1. LSH

- **Recall rises steadily with every doubling of bits:**
  - random: 0.021 → 0.098 → 0.193 → 0.321 → 0.472;
  - FinanceBench: 0.314 → 0.487 → 0.613 → 0.699 → 0.791.
  - *Each bit is one random hyperplane, and two vectors at angle θ share a bit with probability 1 − θ/π. Hamming distance is therefore a noisy estimate of the angle, and the noise shrinks as bits are added.*
- **Cost grows linearly with bits.** Size doubles with each doubling (3.43 → 54.83 MB), and latency grows about 12–14× from 256 to 4096 bits (clustered 0.0079 → 0.107s).
- **Cheapest build of all three methods:** 0.1–6.9s, with no training. *The hyperplanes are random and don't depend on the data.*
- **Never larger than Flat at d=384.** 4096 bits is 2.7× smaller than Flat (54.8 vs 146.5 MB). *A raw vector takes 384 × 32 = 12,288 bits.*
- **Weak on synthetic data:** best 0.472 on random and 0.166 on clustered, against 0.791 on FinanceBench. §5 gives the reason.

## 2. PQ

- **Recall is governed by dimensions per subspace (d/m).** With nbits=10, d/m = 48 → 24 → 12 → 6 → 2 gives:
  - random: 0.021 → 0.050 → 0.175 → 0.347 → 0.853;
  - FinanceBench: 0.236 → 0.384 → 0.553 → 0.755 → 0.957.
  - *For a fixed number of centroids, quantisation distortion grows with subspace dimension (∝ k^(−2/dsub)). 1,024 centroids cover 2 dimensions densely but 48 dimensions very sparsely.*
- **`nbits` matters much less than `m`.** At m=32, going from 64 to 1,024 centroids (nbits 6 → 10) moves random recall only from 0.065 to 0.175. Going from m=32 to m=192 at the same nbits moves it from 0.175 to 0.853.
- **Largest compression of any method:** 6–62× smaller than Flat (2.4–25.3 MB). *Each vector is stored as m small integers instead of 384 floats.*
- **`nbits=8` is the fastest PQ setting on every dataset:** 8.6–10.1× faster than Flat, against about 2× at 6 bits and 1.3–1.5× at 10 bits. *8-bit codes are exactly one byte, so lookups are byte-aligned. 6- and 10-bit codes have to be unpacked from shared bytes.*
- **At high recall PQ is slower than exact search.** m=192 runs at 0.17–0.23× Flat's speed on every dataset. *Plain PQ still scans every code (O(n)), doing m table lookups per vector, while Flat uses vectorised matrix multiplication. This is what motivates IVF (§7).*
- **Most expensive build, growing steeply with nbits.** At m=32 it takes 2.3s → 30.6s → 202.5s for nbits 6 → 8 → 10. *k-means cost scales with the number of centroids, 2^nbits.*
- **Build time does not increase steadily with m.** Random takes 65 / 149 / 202 / 38 / 51s for m = 8 / 16 / 32 / 64 / 192, and the same pattern appears on all three datasets. This is reported as observed; it is likely an implementation effect.
- **PQ needs enough training data.** FAISS warns below about 39 training points per centroid (about 40,000 for nbits=10). The first FinanceBench corpus (1,198 cited passages) could only support nbits ≤ 4, which is why full filings were used. *LSH and HNSW don't have this dependence on data.*
- **Scoring choice: L2 on the normalised vectors, not inner product.** Both rank identically on exact unit vectors, but not on PQ's compressed ones.

  | recall@10, m=192 / m=64 (nbits=10) | inner product | L2 (used) |
  |---|---|---|
  | random | **0.901** / **0.452** | 0.853 / 0.347 |
  | clustered | 0.491 / 0.023 | **0.851** / **0.365** |
  | FinanceBench | 0.955 / 0.727 | **0.957** / **0.755** |

  - L2 is at least as good at all 7 configs on clustered data (up to +0.36) and on FinanceBench (up to +0.05). Inner product is better only on random data (up to +0.10).
  - *Why:* compression makes a vector's length drift by a small δ.
    - Inner product scores cos·(1+δ), so its error grows with cos.
    - L2 scores 2 − 2cos + 2δ(1 − cos), so its error grows with 1 − cos.
    - Inner product is therefore more robust when the true neighbours have cos < 0.5, and L2 when cos > 0.5.
    - Random data's neighbours sit at cos ≈ 0.2, so inner product wins there. Clustered data's sit at cos ≈ 0.98, so inner product's error is about 50× larger and recall collapses.
  - L2 was chosen because it is PQ's native metric and never collapses.

## 3. HNSW

- **`efSearch` tunes accuracy at search time, with no rebuild.** One FinanceBench index spans 0.788 → 0.990 recall as efSearch goes 16 → 256. *efSearch is the size of the candidate list kept during the greedy graph walk. Neither PQ nor LSH can change accuracy without a rebuild.*
- **Diminishing returns in efSearch.** On FinanceBench, going 128 → 256 gains only +0.009 recall for 1.6× the latency.
- **`M` trades memory for recall.** At efSearch=64:
  - FinanceBench: M = 8 / 16 / 32 / 64 gives 0.808 / 0.887 / 0.937 / 0.964;
  - clustered: 0.455 / 0.634 / 0.818 / 0.895.
- **Always larger than Flat**, by 5% (M=8) to 34% (M=64).
  - The overhead is 81 / 144 / 272 / 528 bytes per vector, roughly 2·M·4 bytes.
  - *HNSW stores the raw vectors plus M neighbour IDs per node (2M at the base layer). That overhead doesn't depend on d.*
- **Fastest method at high recall:**
  - FinanceBench: 0.981 recall at 19× faster than Flat, and 0.990 at 12×;
  - clustered: 0.913 at 13×, and 0.981 at 3.3×.
  - *The greedy walk visits only a small fraction of the database.*
- **Worst on random data:** best 0.403, and still climbing (+0.144 from efSearch 128 → 256), so not yet saturated. *See §5: random data gives the walk nothing to follow.*
- **Builds faster on easier data.** At M=32: FinanceBench 10.9s, clustered 19.3s, random 25.7s. *Building the graph is itself a series of searches, so data that is easy to search is also quick to index.*

## 4. Method comparison

- **Each method wins a different resource:**

  | | memory | search speed at high recall | build / training |
  |---|---|---|---|
  | LSH | medium (3.4–57 MB) | fast, but low recall | **cheapest (≤ 7s, no training)** |
  | PQ | **smallest (2.4–25 MB)** | slower than Flat | most expensive (up to 217s) |
  | HNSW | largest (154–205 MB, more than Flat) | **fastest (0.99 recall at 12× Flat)** | moderate (5–35s) |

- **At equal memory, LSH is faster and PQ is more accurate.** Both use 32 bytes per vector (LSH 256 bits vs PQ m=32, nbits=8). On FinanceBench: recall 0.314 vs 0.489, latency 0.015s vs 0.064s. *Hamming distance takes a few XOR + popcount instructions per vector; PQ does m table lookups but reconstructs a much better distance estimate.*
- **Only HNSW reaches high recall while staying faster than Flat.**
  - HNSW reaches ≥ 0.95 at 12–19× Flat's speed on FinanceBench.
  - PQ reaches 0.957 only at m=192, running at 0.17× Flat's speed.
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
  | HNSW (M=32, efSearch=256) | 0.403 | 0.981 | **0.990** |

- **All three methods do best on real data**, even though every dataset has 384 dimensions. *Text embeddings lie near a much lower-dimensional structure (companies, statement types, topics). Random data genuinely uses all 384 dimensions: 323 of them are needed to keep 90% of the variance (eda4).*
- **HNSW depends on global contrast.**
  - On random data the average point is only 1.13× as far away as the nearest one; on clustered data it is 6.1×.
  - HNSW recall follows: 0.403 vs 0.981.
  - *Each greedy hop moves to the neighbour closest to the query. When every point is almost equally far away, hops make no progress.*
- **LSH depends on angle gaps, and clusters shrink them.**
  - Inside a cluster, the 10th and 1000th neighbours differ by only 1.0°; on random data they differ by 4.1°.
  - *Each bit agrees with probability 1 − θ/π, so the clustered gap carries about 4× less information per bit.* Taking the noise of each bit into account, clustered data needs about 4–5× more bits to do as well.
  - The data bears this out: clustered at 4096 bits (0.166) sits between random at 512 bits (0.098) and at 1024 bits (0.193).
  - So data that is easy to search by distance can still be hard for LSH.
- **PQ is indifferent to cluster structure:** random 0.853, clustered 0.851 at m=192. *PQ has to rank the neighbours inside the query's own cluster, and that local problem is about as hard in both datasets (distance ratio between the 10th and 100th neighbour: 0.981 vs 0.966).*
- **On random data LSH currently beats HNSW (0.472 vs 0.403).** This is tentative, because HNSW had not saturated at efSearch=256 and M=64 was only tested at efSearch=64.

## 6. Conclusions: which method when

| situation | choose | because |
|---|---|---|
| memory is the binding constraint | **PQ** (nbits=8 for speed) | 6–62× compression |
| latency-critical, high recall needed | **HNSW** | ≥ 0.95 recall at 12–19× Flat's speed on real data |
| data changes constantly / no training data / cheapest comparisons | **LSH** | no training, builds in seconds, Hamming distance |
| data is isotropic and high-dimensional | none performs well | curse of dimensionality: reconsider the representation |

## 7. Improvements, limitations, open items

- **Improvements:**
  - **Implemented:** PQ scoring with L2 instead of inner product (§2). Recall rose by up to +0.36 on clustered data and +0.05 on FinanceBench.
  - **IVF (discussed):** partition the database into cells and probe only a few, making PQ's scan sub-linear. This addresses PQ being slower than Flat.
  - **OPQ (discussed):** learn a rotation before quantising, so correlated dimensions share a subspace.
  - **HNSW + PQ (discussed):** store PQ codes in the graph to cut HNSW's memory.
- **Limitations:**
  - CPU-only laptop timings, with run-to-run noise.
  - One embedding model, and 150 FinanceBench queries.
  - The synthetic data is Gaussian.
  - Cosine discards vector length, which carries information in the raw clustered data (top-10 overlap between L2 and cosine on the raw vectors is 0.80). Conclusions may not transfer to tasks where vector length matters.
  - FAISS `IndexLSH` scans every code (no hash-bucket lookup), so LSH is measured as compact codes rather than bucketed retrieval.
  - Parameters were swept one at a time around M=32 / efSearch=64 and m=32 / nbits=10, not as a full grid.
- **Open items:**
  - A full parameter grid, especially HNSW M × efSearch on random data, to settle LSH vs HNSW.
  - FinanceBench EDA: check that its neighbour similarity at the top-10 boundary exceeds 0.5, as the PQ scoring result implies.
