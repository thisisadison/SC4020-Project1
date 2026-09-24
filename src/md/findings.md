# Findings — d = 384

All datasets d=384. Synthetic: 100,000 vectors, 100 queries. FinanceBench: 104,169 passages, 150 questions.
Speedup = Flat latency / method latency, measured in the same run, so database size and query count cancel out.

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