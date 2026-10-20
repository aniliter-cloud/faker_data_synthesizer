# GPU K-Means: Atomic, Reduction and Fused Assignment

We compare a CPU baseline with CUDA implementations of Lloyd's K-Means using
global atomics, shared-memory reductions and a custom fused dot-score/argmin
kernel. Our dimension sweep retains every row and the dataset's K while selecting
the first **2, 4 or 9 features**.

- [Git repository](https://github.com/aniliter-cloud/kmeans-gpu-grp-project)
- [Benchmark results and profiling files](https://drive.google.com/drive/folders/1QWPLpS4KsT1SCrnX12BwzoZomqfjtgBh?usp=drive_link)
- [Two-page assignment report](KMEANS_PROJECT_REPORT.pdf)

## 1. Generate the data locally

```bash
git clone https://github.com/aniliter-cloud/kmeans-gpu-grp-project.git
cd kmeans-gpu-grp-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.
The generator and preparation script only require Faker, NumPy and pandas; a GPU
is not needed for this step. The checked-in dependency snapshot differs from the
recorded benchmark environment; see [reproducibility notes](docs/EXPERIMENTS_AND_RESULTS.md#environment-and-reproducibility).

`faker_data_synthesizer.py` creates clustered float64 features and Faker UUIDs.
With no arguments it generates **nine large datasets**:

| N (rows) | K values |
|---:|---|
| 1,000,000 | 3, 5, 7 |
| 5,000,000 | 7, 9, 11 |
| 10,000,000 | 13, 15, 17 |

To reproduce the complete **18-dataset** benchmark collection, also generate
the three smaller row groups:

```bash
python faker_data_synthesizer.py
python faker_data_synthesizer.py --rows 10000 --cols 10 --clusters 3 5 7
python faker_data_synthesizer.py --rows 50000 --cols 10 --clusters 7 9 11
python faker_data_synthesizer.py --rows 100000 --cols 10 --clusters 13 15 17
python prepare_initial_centroids.py --data-root .
```

Generation is sequential and the larger CSVs need substantial disk space and
memory. `--cols 10` means ID plus nine features; the appended `true_label` makes
11 physical CSV columns. Older supplied files may omit `true_label`.

The preparation script samples K rows with seed 42 once in full 9D and writes
canonical initial-centroid CSVs. It reuses existing valid references without
overwriting them. Generate new experiments in a separate data root rather than
mixing regenerated data with old canonical files or completed results.

## 2. Upload to Google Drive

Upload the generated `rows*` folders and `results_gpu_atomic_v3` together:

```text
MyDrive/kmeans_data/
  rows10000/
  rows50000/
  rows100000/
  rows1000000/
  rows5000000/
  rows10000000/
  results_gpu_atomic_v3/
    initial_centroids_synthetic_data_<N>_rows_10_cols_<K>_clusters.csv
```

Upload completed CSVs without changing their names or row order. The notebooks
also recognize `results_gpu_atomic_v3 2` as an optional sibling folder when
reference files were downloaded in multiple parts.

## 3. Run the notebooks in Colab

| Notebook | Purpose |
|---|---|
| `kmeans_cpu_baseline_V3_ALIGNED.ipynb` | CPU Numba baseline and repeated fixed-iteration timing |
| `kmeans_parallel_atomic_V3_ALIGNED.ipynb` | CUDA assignment and global-atomic sums/counts |
| `kmeans_parallel_reduction_V3_ALIGNED.ipynb` | CUDA assignment and shared-memory scan reductions |
| `kmeans_gemm_fused_V3_ALIGNED.ipynb` | Fused dot-score/argmin assignment and atomic accumulation |

Open a notebook in Colab, mount Drive, and confirm its data root points to
`/content/drive/MyDrive/kmeans_data`. Use a CPU runtime for the baseline and an
NVIDIA GPU runtime for the CUDA notebooks. The GPU notebooks require Numba CUDA;
the fused notebook also requires CuPy. Confirm the GPU with `!nvidia-smi`.

Keep the full-run settings:

```python
FEATURE_DIMS = [2, 4, 9]
SWEEP_CASES = None
```

Run cells from the top. Atomic/Reduction also default to `RUN_MODE = "all"`.
The prepared canonical files allow the notebooks to run in any order. We never
cluster ID or `true_label`; each D uses matching columns from the same canonical
centroid rows. No N value needs to be entered for the batch.

In the fused notebook, Parts A-D demonstrate one case; their `ROWS`, `K` and
`FEATURE_DIM` settings do not restrict Part E. Part E contains the standard
18-dataset batch list. Generate the full collection above before running it.
`SUM_METHOD = "atomic"` is the audited sweep variant; `"shared"` writes a
separate experimental result folder.

## 4. Results and comparison

Each implementation writes a new `*_dimension_sweep` result folder with a
summary and `_D2`, `_D4` or `_D9` per-case files. The full collection produces
**54 cases per implementation**. Keep the environment JSON, histories and
assignment subfolders when downloading; Drive may split large downloads.
The CPU packaging cell now copies staged assignments before creating a recursive
ZIP. The fused notebook currently saves centroids/sizes and histories, but does
not export per-point assignments.

Compare the same source dataset, N, K and D, and check initial-centroid hashes.
CPU performs a final nearest-centroid reassignment after stopping; GPU files
retain labels from the last iteration. This explains small differences in
exported cluster sizes and WCSS despite matching iterative centroids. CPU
fixed-iteration medians and GPU convergence times are different measurements.

We validated 54 final cases per method. Fused assignment is not consistently
fastest across all shapes. See [experiments and results](docs/EXPERIMENTS_AND_RESULTS.md)
for the timings, numerical checks and remaining download gaps.

## Design and profiling

- [Implementation design](docs/IMPLEMENTATION_DESIGN.md)
- [Experiments, results and reproducibility](docs/EXPERIMENTS_AND_RESULTS.md)
- [Nsight profiling evidence and procedure](docs/PROFILING.md)

The fused iterative path uses a custom CUDA kernel based on the GEMM distance
identity; its standalone CuPy experiment invokes cuBLAS separately. Nsight kernel
times and complete convergence timings must be reported with their own scopes.
