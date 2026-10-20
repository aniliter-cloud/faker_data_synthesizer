# Implementation design

## Objective and scope

We implement Lloyd's K-Means with a CPU baseline and two required CUDA centroid
accumulation strategies, then explore fused assignment using the GEMM distance
identity. We vary N, K and D and compare correctness before interpreting runtime.
The four `V3_ALIGNED` notebooks are our dimension-sweep entry points. Earlier
notebooks remain historical experiments; they are not interchangeable benchmark
drivers.

## Data model and generation

`faker_data_synthesizer.py` creates features with NumPy and IDs with Faker.
For each file it initializes `default_rng(42)`, samples K centers with coordinates
uniformly in **[-100, 1000]**, assigns one standard deviation in [1,10] per center,
and chooses a generating cluster for each point. Features are the chosen center
plus independent normal noise scaled by that cluster's standard deviation.
These generating centers are separate from the K-Means initial centroids.

The generator returns ID and `col_1,...,col_9`; `save()` appends `true_label`.
The filenames retain `10_cols`, meaning ID plus nine features, even when the
actual CSV has 11 columns after adding the label. We exclude both metadata
columns from clustering. Older supplied datasets may have no true-label column.
We do not claim adjusted Rand scores or recovery of all generating clusters from
WCSS alone.

Features are binary float64 values written with 16 decimal places. The SQL DDL
printed by the generator is a schema illustration, not the in-memory dtype.
Padding decimal places does not create additional floating-point precision.
Seeded generation is reproducible within a controlled dependency environment;
we do not promise byte-identical CSVs across arbitrary library versions.

The generator's default plan contains N=1M,5M,10M, with three K values per N.
The smaller N=10k,50k,100k groups require the additional commands in the README.
Together these give 18 source files. The default plan does not silently create
all 18 files.

## Canonical initialization

We require one full-9D canonical matrix per source file, formatted as
`cluster,col_1,...,col_9` with K rows. `prepare_initial_centroids.py` makes this
dependency explicit for fresh datasets:

1. Discover the expected source CSVs in `rows*` directories.
2. Draw K distinct zero-based row indices with `default_rng(42)`.
3. Stream all nine feature columns in bounded chunks, retaining the RNG's row
   order in the resulting K×9 matrix.
4. Verify the declared row count and finite values, then create the canonical
   CSV exclusively so an existing reference is not replaced.
5. Record preparation metadata and a CSV checksum. Existing references in the
   primary or optional second folder are schema-checked and reused.

We prepare canonical files before uploading to Drive. Each notebook reads the
same reference and records the SHA-256 of the contiguous parsed float64 matrix
actually used for that case. A CSV checksum and a parsed-matrix hash are
different checks. Runtime dependency versions belong with the results.

## Dimension selection

`FEATURE_DIMS = [2,4,9]` selects nested feature subsets by name, not incidental
CSV order. D=2 uses `col_1,col_2`; D=4 adds `col_3,col_4`; D=9 uses all features.
We select identical columns from the canonical matrix, retaining all points and
the same K. We do not use PCA, sample fewer rows or initialize each D separately.
`SWEEP_CASES = None` runs the full selected collection. N and K come from the
existing dataset selection; they are not required user inputs for the batch.

## Lloyd iteration and stopping

For each point x and centroid c, our direct assignment minimizes
`sum_j (x_j-c_j)^2`. We sum coordinates and counts for each assigned cluster and
set each nonempty centroid to its mean. An empty cluster retains its previous
centroid. We initialize labels to -1, so every point changes on iteration one.

We use float64 features and centroid sums, a 400-iteration limit, and stop when
either condition is met:

- Maximum Euclidean centroid displacement is below `1e-4`.
- Changed-assignment fraction is at most `1e-4` for three consecutive iterations.

The second condition allows a small nonzero number of changes. It is not an
exact-zero label-stability test. We preserve this contract across the notebooks.

## CPU baseline

The primary CPU path uses Numba-parallel assignment/partial accumulation followed
by centroid updates. A NumPy fallback is available. We compare the implementation
against a direct float64 reference before benchmarking. Two measurements are
saved separately: repeated fixed-iteration timing with convergence disabled, and
a run to the aligned stopping condition. Thread-count selection and timing
settings are saved in the CPU environment and summary.

After the timed convergence loop, CPU assigns points once more to the final
centroids and uses those labels for exported sizes, distances and WCSS. This
extra assignment is not another centroid update and is not in the convergence
timer. It is a material output-stage difference from the GPU notebooks.

## GPU Atomic

We assign one CUDA thread per point. Threads cooperatively cache centroids in
shared memory and reduce label-change flags per block before contributing to
the global changed counter. Each point then uses global atomics to add its
coordinates and count to the assigned cluster. A small GPU kernel divides sums
by counts and retains old centroids for empty clusters.

This avoids concurrent-write races but can create contention when many points
contribute to relatively few centroid coordinates. Nsight timing supports the
importance of accumulation in the profiled workloads; it does not establish a
hardware-counter diagnosis of that contention.

## GPU Reduction

We retain the same assignment kernel and stopping contract. For centroid sums,
each block owns a `(cluster, feature)` pair. Threads scan input rows in strides,
accumulate contributions locally and reduce them in shared memory. A separate
per-cluster reduction counts points. The mean-update kernel matches Atomic.

The strategy trades repeated scans against per-point global atomic updates.
It can improve some workloads and lose on others. The dimension sweep measures
that tradeoff rather than assuming reductions are always faster.

## Fused dot-score/argmin assignment

We use `||x-c||² = ||x||² - 2(x·c) + ||c||²`. The point norm is independent of
the candidate cluster, so assignment minimizes `||c||² - 2(x·c)`. The iterative
kernel reads transposed D×N data, caches centroids, computes dot scores and keeps
only the best label. It does not materialize the full N×K score matrix.

This is a **custom fused CUDA assignment kernel using the GEMM identity**.
It is not a cuBLAS GEMM call on every K-Means iteration. The notebook separately
includes naive/tiled dot-product kernels and a CuPy/cuBLAS assignment experiment.
Their measurement scopes differ: dot-product-only tests exclude score formation
and argmin, while complete assignment includes those operations.

Fused clustering supports global atomic accumulation and block-local shared
atomic accumulation followed by global atomics. The latter remains an atomic
method, distinct from the scan-based Reduction notebook. We download sums and
counts and update centroids on the CPU, then upload the state required by the
next assignment. The audited dimension sweep uses global atomic sums; shared
sums have separate profiler evidence and a separate optional output folder.

## Timing boundaries and state

We exclude data loading and JIT warm-up from convergence timing. GPU feature
uploads precede the timed iterative loop. Atomic/Reduction use persistent device
arrays and exclude their final assignment download from the timer. Fused creates
and transfers several small arrays during each iteration, updates means on the
host and returns downloaded labels within its timed call. Those costs remain
part of its measured implementation.

CPU fixed-iteration median times, convergence wall times, profiler kernel sums
and first-to-last-kernel windows are separate measurements. We label each scope
and never add overlapping synchronization durations to GPU execution time.

## Persistence and export

Each case uses `<source_tag>_D<dimension>` for result identity, resume checks and
artifact names. Summaries record feature names, dimensions, initialization hash,
iterations, reason, timing, WCSS, centroids and cluster sizes, with a different
lowercase schema retained in the CPU summary. Per-case final-result CSVs use a
common schema. We preserve the original dataset filename when locating the
full-9D canonical reference.

Atomic/Reduction save assignment CSVs, histories and centroid tables. CPU saves
assignments to a local staging directory on Colab; its revised export cell copies
them to the output directory before recursively zipping results. This fixes the
older package that excluded assignments/history. Fused currently saves final
centroids and sizes in JSON fields and histories, without per-point CSV exports.
This cleanup does not change its clustering loop or add missing historical data.

## Next implementation improvements

We would standardize the final label-export convention across methods, retain
fused state and mean updates on the GPU, reuse buffers, add application NVTX
ranges, and measure repeated full-run timing distributions. These are future
changes, not features claimed by the present benchmark results.
