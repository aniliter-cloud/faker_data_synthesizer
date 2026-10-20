# Experiments, results and reproducibility

## Experiment matrix

We evaluate the following source files, selecting D=2,4,9 from every file:

| N | K values |
|---:|---|
| 10,000 | 3, 5, 7 |
| 50,000 | 7, 9, 11 |
| 100,000 | 13, 15, 17 |
| 1,000,000 | 3, 5, 7 |
| 5,000,000 | 7, 9, 11 |
| 10,000,000 | 13, 15, 17 |

This is 18 source datasets, 54 cases per method, and **216 saved final runs**
across CPU, Atomic, Reduction and Fused. N and K remain fixed within a source
dataset. We use all its rows at each D.

We match results using source dataset, N, K and D and verify initialization hashes
and selected feature names. A dimension sweep changes the clustering problem:
iteration counts and WCSS need not be equal across D. We do not rank clustering
quality by comparing raw WCSS across different feature spaces.

## Environment and reproducibility

The saved GPU environments report Tesla T4, compute capability 7.5, Python
3.13.15, NumPy 2.1.3, pandas 2.2.3 and Numba 0.61.2. CPU reports a Colab Xeon
at 2.20 GHz, two logical cores, Python 3.13.16 and the same NumPy/pandas/Numba
versions. CPU tries the available thread-count candidates and records its choice.

The existing `requirements.txt` is a separate dependency snapshot: Faker 40.40.0,
NumPy 2.5.3, pandas 3.0.6 and Numba 0.65.0 among its pins. We have not relabeled
those pins as the versions used for the reported measurements. For comparison
with our results, preserve the supplied CSVs and canonical references, record the
actual environment, and verify hashes. Re-running generation under different
dependencies does not guarantee byte-identical CSVs.

Our main defaults are float64, seed 42 for canonical row selection, movement
tolerance 1e-4, changed-assignment fraction 1e-4 for three consecutive iterations,
and a 400-iteration limit. Warm-up and data loading are excluded from the timed
convergence loop. See [design](IMPLEMENTATION_DESIGN.md) for the remaining timing
boundary differences.

## Validation results

- All four methods have **54/54 final-result cases**, with 18 at each D.
- Initial-centroid matrix hashes agree across all methods in **54/54 cases**.
- Iteration counts, convergence reasons and final changed-assignment counts agree
  across methods in **54/54 cases**. Every saved run reports convergence.
- The maximum absolute final-centroid coordinate difference across method pairs
  is below **7.69e-11**.
- All three GPU methods have identical cluster-size vectors and changed-count
  histories in **54/54 cases**. Maximum relative GPU WCSS difference is below
  **2.79e-15**.
- All **41 available CPU histories** match the GPU changed-count sequences.
  Thirteen CPU histories were absent from the local export, so they were not
  assumed to have been checked.
- All available histories satisfy the recorded stopping contract. The largest
  iteration count is 372. Each method has 39 assignment-stability stops,
  13 movement-tolerance stops and 2 meeting both conditions.
- The D=9 numerical results reproduce the earlier fixed-9D outputs wherever
  available: 18 Atomic, 18 Reduction and 9 Fused cases match iteration counts,
  sizes and WCSS to numerical tolerance.

These checks cover all final-result records, 203 available histories, 162 saved
projected initial matrices, and their metadata. Matching cluster counts alone
would not prove per-point label equality; our claims distinguish that limitation.

## CPU final-label convention

CPU recomputes nearest-centroid labels after stopping; GPU exports retain the
labels from the last assignment before its final mean update. Consequently,
CPU/GPU exported cluster sizes agree in 19/54 cases and differ in 35/54. This is
an output-stage difference despite closely matching iterative centroids. We do
not describe every WCSS difference solely as floating-point rounding.

The maximum relative CPU/GPU WCSS difference is about **5.60e-9**. A strict 1e-9
relative WCSS test passes in 50/54 cases; a looser tolerance is not a substitute
for explaining the label convention.

We independently streamed source features and Atomic/Reduction assignment files
for two complete cases and reproduced the CPU output through final reassignment:

| N | K | D | Labels changed after stopping | Atomic/Reduction label mismatches |
|---:|---:|---:|---:|---:|
| 100,000 | 13 | 4 | 7 | 0 |
| 10,000,000 | 17 | 2 | 909 | 0 |

All 10.1M rows in these two cases had matching source/export ID order. Recomputed
sizes and WCSS match the corresponding saved CPU and GPU values. The maximum
saved-distance discrepancy was below 3.24e-13. This is not an exhaustive
row-by-row validation of all 54 cases or of the unavailable Fused label arrays.

For strict exported-label comparison, we would apply the same post-stop
nearest-centroid assignment to every method outside the iteration timer and keep
the original stopping history. Existing final centroids allow that postprocessing
without rerunning the Lloyd iterations. The current cleanup preserves the
historical label conventions and numerical results.

## Convergence runtime example

N=10,000,000, K=17. Values are recorded convergence seconds, excluding warm-up:

| D | Iterations | CPU | Atomic | Reduction | Fused |
|---:|---:|---:|---:|---:|---:|
| 2 | 100 | 80.032 | 2.995 | 4.674 | 2.951 |
| 4 | 104 | 107.218 | 3.972 | 5.957 | 3.640 |
| 9 | 183 | 369.043 | 14.281 | 17.944 | 11.243 |

For D=9 in this example, Fused takes about 1.27x less time than Atomic, 1.60x
less time than Reduction, and 32.82x less time than the CPU convergence run.
These are ratios of recorded timings with their implementation-specific scopes,
not repeated-run confidence estimates or CPU fixed-iteration benchmark ratios.

Across all cases, Fused is not consistently fastest:

| D | Fused faster than Atomic | Fused faster than Reduction | Fused faster than CPU |
|---:|---:|---:|---:|
| 2 | 3/18 | 12/18 | 13/18 |
| 4 | 4/18 | 6/18 | 15/18 |
| 9 | 8/18 | 4/18 | 15/18 |

Lower D reduces coordinate work, but convergence may increase. For 10M/K=13,
we observe 231 iterations at D=2, 42 at D=4 and 185 at D=9. Per-iteration cost
and total runtime should therefore be discussed separately. Launch, allocation,
transfer and CPU-update overhead are plausible contributors to performance
variation, not isolated causal conclusions from the sweep timings.

## Local export completeness

Our numerical run coverage is complete; our reviewed local downloads have gaps:

| Method | Final cases | Selected summary | Histories | Assignment CSVs |
|---|---:|---:|---:|---:|
| CPU | 54 | 54 | 41 | 0 |
| Atomic | 54 | 54 | 54 | 54 |
| Reduction | 54 | 53 | 54 | 53 |
| Fused | 54 | 54 | 54 | 0 |

- The complete CPU summary is in `cpu_baseline_results (5)`; the separate CPU
  sweep folder contains an earlier 41-case snapshot. Overlapping records agree.
- Reduction's downloaded summary and assignments omit 10M/K=17/D=9. That case's
  final-result CSV, centroid table, initial matrix and 183-row history exist.
- CPU lacks all nine 10M histories, all three 5M/K=11 histories, and 5M/K=9/D=9.
  Its old ZIP included top-level files only and was created before staged
  assignments were copied. The revised packaging cell fixes future exports;
  it cannot recover files absent from the current download.
- Fused currently does not export per-point labels. Its final centroids and
  cluster sizes are stored as JSON in final-result CSVs.

Retrieve the missing exports from Drive/runtime if present. Reconstruct the
Reduction summary from all 54 final-result CSVs if needed. These export gaps do
not imply failed iterations, but they should not be hidden in a submission.

## Evidence and source locations

[Shared results folder](https://drive.google.com/drive/folders/1QWPLpS4KsT1SCrnX12BwzoZomqfjtgBh?usp=drive_link)
contains the user-specified result location. Our quantitative review used the
locally supplied result copies, not an independent verification of that folder's
remote completeness or access permissions.

We combined Atomic/Reduction download parts by artifact name. Eighty-four
duplicate artifacts were byte-identical. Primary summaries were the complete CPU
package, Atomic sweep folder, Reduction sweep part 4, and Fused sweep folder.
We also checked per-case CSVs, histories, projected matrices and environment JSON.
The separate Nsight workloads are documented in [profiling](PROFILING.md).
