# Nsight Systems profiling

## Captures and scope

We reviewed six supplied Nsight Systems SQLite exports for Tesla T4: Atomic,
Reduction and files named `gemm_kmeans`, each at **500k/K=11** and **1M/K=17**.
Shapes and transfer volumes are consistent with D=9 float64 inputs. These are
separate workloads from the 18-file dimension sweep, and the traces do not prove
input contents, canonical hashes or final clustering correctness.

All six exports passed SQLite `quick_check` and contained CUDA kernel, API and
copy events for a main process. Export metadata identifies Nsight Systems
2026.5.1.161. We removed identifiable warm-up and selected full-size phases using
launch geometry and kernel ordering. The selected phases contain 60 full-size
iterations for 500k/K=11 and 129 for 1M/K=17.

The original standalone capture drivers are not supplied. Application NVTX
ranges do not identify the phase boundaries; we infer those boundaries from
events and matching notebook source. The capture names alone are not enough to
define the experiment.

## Mean GPU kernel cost per full clustering iteration

Milliseconds below are sums of selected GPU kernel durations divided by their
iteration counts. They exclude Python orchestration, CPU centroid updates,
inter-kernel gaps and operations outside the selected phase.

| Phase | 500k / K=11 | 1M / K=17 |
|---|---:|---:|
| Atomic | 5.368 | 7.932 |
| Reduction | 4.564 | 7.451 |
| Fused + global atomic sums | 3.694 | 5.554 |
| Fused + shared atomic sums | 2.337 | 5.366 |

Atomic accumulation accounts for approximately 61.7% and 54.4% of selected GPU
kernel time in these captures. The final mean-update kernel is negligible.
Reduction reduces selected kernel cost relative to Atomic in both captures.
Fused assignment also takes less kernel time in these profiles, and shared
atomic accumulation improves the fused pipeline most clearly at 500k/K=11.

These are not isolated arithmetic comparisons: data layout, accumulation and
host orchestration differ. In particular, Fused computes means on the CPU, so
its GPU totals exclude work that Atomic/Reduction perform on the GPU. The table
does not establish a general end-to-end speed-up or replace sweep timings.

## What the `gemm_kmeans` captures contain

They match the fused notebook's mixed experiment sequence: standalone naive and
tiled dot-product matrices, custom fused assignment, a CuPy assignment pipeline,
small clustering warm-up, then full fused clustering with two accumulation modes.
The 11 cuBLAS GEMM calls correspond to one warm-up and ten standalone assignment
repetitions; cuBLAS is not called on every full clustering iteration.

After excluding the first warm-up, median summed GPU kernel costs for complete
assignment are:

| Workload | Custom fused assignment | CuPy complete assignment |
|---|---:|---:|
| 500k / K=11 | 0.998 ms | 8.962 ms |
| 1M / K=17 | 1.793 ms | 13.328 ms |

These imply about 9.0x and 7.4x lower kernel cost for custom fused assignment in
those standalone experiments. They do not compare whole K-Means convergence.
Naive/tiled dot-product-only timings exclude score construction and argmin and
must not be presented as complete assignment timings.

The tiled 16x16 dot-product kernel is slower than the naive dot-product kernel
in both captures. Padding D=9 to the tile width, synchronization overhead and
strided access are code-based hypotheses to investigate, not measured
hardware-counter explanations.

## Orchestration, synchronization and transfers

Fused clustering allocates/uploads centroids, norms, counters, sums and counts
on each iteration, downloads results, and updates means on the host. These
repeated small synchronous operations and allocations are concrete targets for
buffer reuse and a device-resident update path.

Synchronization API durations overlap GPU execution. Adding them to kernel time
would double-count work. Blocking copy API latency can include waiting for prior
kernels; it is not necessarily physical transfer time. Complete capture windows
include startup, compilation, validation and separate experiments and are not
clean benchmark timer totals.

The exports do not supply measured occupancy, bandwidth, cache hit rate or warp
stall counters. We do not claim a kernel is compute-bound, bandwidth-bound or
occupancy-limited from these timings alone. A VS Code extension or a trace GUI
is also not evidence that the `nsys` capture CLI exists in a given terminal.

## Reproducing a focused capture

Run capture commands on a CUDA-capable target with the Nsight Systems CLI
installed, not in a Mac terminal without `nsys`. Check the target first:

```bash
nsys --version
nsys status --help
```

Create a standalone driver for one implementation and one workload. Keep data
loading and compilation separate from the iterative measurement, add application
NVTX ranges for warm-up, assignment, accumulation, update and reporting, and save
a manifest containing N/D/K, input/reference hashes, source revision, environment
and final convergence result. The historical `run_*.py` driver names in the
captures are not supplied by this repository refresh.

For a driver named `profile_case.py` that has been created and verified:

```bash
nsys profile --trace=cuda,nvtx --sample=none --output=case_profile python profile_case.py
nsys export --type=sqlite --output=case_profile.sqlite case_profile.nsys-rep
nsys stats case_profile.nsys-rep
```

Those commands illustrate collection/export; they do not implement the missing
driver. Inspect supported options with the installed CLI's `--help`, and use
capture-range controls when the driver defines the corresponding NVTX/profiler
boundaries. A broad `--trace=cuda,nvtx` capture still includes setup unless its
scope is explicitly controlled.

For fair comparisons we would capture one case at a time, verify matching
initialization/results, collect repeated wall-time runs separately, and retain
GPU execution, host API duration and complete loop wall time as distinct metrics.

## References

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA post-collection analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- Supplied `.nsys-rep` and six `.sqlite` exports under `profiling-report/`.

We read the SQLite exports without modifying the source profiles. The
user-specified [shared results folder](https://drive.google.com/drive/folders/1QWPLpS4KsT1SCrnX12BwzoZomqfjtgBh?usp=drive_link)
is the external location for project evidence; remote access/completeness was
not independently established by the local profiling review.
