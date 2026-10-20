"""Prepare shared full-9D initial centroids before uploading data to Colab."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PATTERN = re.compile(r"^synthetic_data_(\d+)_rows_10_cols_(\d+)_clusters\.csv$")
FEATURES = [f"col_{i}" for i in range(1, 10)]


def validate_existing(path, clusters):
    """Check the canonical schema without replacing an existing experiment."""
    frame = pd.read_csv(path)
    if not set(FEATURES).issubset(frame.columns):
        raise ValueError(f"{path}: expected all nine feature columns")
    matrix = frame[FEATURES].to_numpy(dtype=np.float64)
    if matrix.shape != (clusters, 9) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: expected a finite {(clusters, 9)} centroid matrix")
    if "cluster" in frame and not np.array_equal(frame["cluster"], np.arange(clusters)):
        raise ValueError(f"{path}: cluster rows must be ordered 0 through K-1")


def prepare_one(source, output_dir, seed=42, chunksize=200_000):
    """Sample rows once in nine dimensions, retaining the RNG's centroid order."""
    match = PATTERN.fullmatch(source.name)
    if match is None:
        raise ValueError(f"Unsupported input filename: {source.name}")
    n, clusters = map(int, match.groups())
    if not 0 < clusters <= n or chunksize < 1:
        raise ValueError("Require 0 < K <= N and a positive chunk size")
    filename = f"initial_centroids_{source.stem}.csv"
    candidates = [output_dir / filename, output_dir.parent / "results_gpu_atomic_v3 2" / filename]
    for candidate in candidates:
        if candidate.exists():
            validate_existing(candidate, clusters)
            return {"dataset": source.stem, "status": "reused", "path": str(candidate)}

    selected = np.random.default_rng(seed).choice(n, clusters, replace=False)
    initial = np.empty((clusters, 9), dtype=np.float64)
    filled = np.zeros(clusters, dtype=bool)
    offset = 0
    for chunk in pd.read_csv(source, usecols=FEATURES, dtype=np.float64, chunksize=chunksize):
        values = chunk[FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: feature values must be finite")
        positions = np.flatnonzero((selected >= offset) & (selected < offset + len(chunk)))
        initial[positions] = values[selected[positions] - offset]
        filled[positions] = True
        offset += len(chunk)
    if offset != n or not filled.all():
        raise ValueError(f"{source}: filename declares {n} rows, found {offset}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    frame = pd.DataFrame(initial, columns=FEATURES)
    frame.insert(0, "cluster", np.arange(clusters))
    with target.open("x") as stream:
        frame.to_csv(stream, index=False)
    return {"dataset": source.stem, "status": "created", "path": str(target),
            "rows": n, "K": clusters, "features": 9, "seed": seed,
            "selected_row_indices_zero_based": selected.tolist(),
            "csv_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    sources = sorted(p for p in root.glob("rows*/synthetic_data_*_rows_10_cols_*_clusters.csv")
                     if PATTERN.fullmatch(p.name))
    if not sources:
        parser.error(f"No matching 9-feature datasets under {root}; run faker_data_synthesizer.py first")
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    output_dir = root / "results_gpu_atomic_v3"
    records = []
    for source in sources:
        record = prepare_one(source, output_dir, args.seed, args.chunksize)
        records.append(record)
        print(f"{record['status']}: {record['path']}")
    manifest = output_dir / "preparation_manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    previous = {r["dataset"]: r for r in existing.get("datasets", [])}
    for record in records:
        if record["status"] == "created" or record["dataset"] not in previous:
            previous[record["dataset"]] = record
    output_dir.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({"numpy": np.__version__, "pandas": pd.__version__,
                                   "datasets": list(previous.values())}, indent=2) + "\n")
    print(f"Ready: {len(records)} canonical files. Upload the rows* and canonical folders together.")


if __name__ == "__main__":
    main()
