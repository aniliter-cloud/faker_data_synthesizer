# Data Synthesizer

Generates reproducible synthetic, clustered datasets with Faker for benchmarking **K-means clustering on GPU**.

## Setup

```bash
python3 -m venv venvds
source venvds/bin/activate          # Windows: venvds\Scripts\Activate.ps1
pip install -r requirements.txt     # or: pip install faker pandas numpy
```

Requires Python 3.10+.

## Run

**Full plan** (9 datasets from `CLUSTER_PLAN`):
```bash
python data_synthesizer.py
```

| Rows    | Clusters (K) |
|---------|--------------|
| 10,000  | 3, 5, 7      |
| 50,000  | 7, 9, 11     |
| 100,000 | 13, 15, 17   |

**Custom run:**
```bash
python faker_data_synthesizer.py --rows 10 --cols 10 --clusters 3 5 7
```

| Argument     | Default | Description                                   |
|--------------|---------|-----------------------------------------------|
| `--rows`     | —       | Rows per dataset. If omitted, runs full plan  |
| `--cols`     | 10      | Total columns, including `ID`                 |
| `--clusters` | 3       | One or more K values; one file per K          |

## Output

Files are saved in a `rows<N>/` folder (created if missing):

```
rows10/
  synthetic_data_10_rows_10_cols_3_clusters.csv
  synthetic_data_10_rows_10_cols_5_clusters.csv
  synthetic_data_10_rows_10_cols_7_clusters.csv
```

### Schema

```sql
CREATE TABLE synthetic_data (
  ID CHAR(32),               -- uuid4 hex via faker.col_id()
  col_1 DECIMAL(32, 16),     -- features col_1 ... col_9
  ...
  col_9 DECIMAL(32, 16),
  true_label INT             -- cluster the row was generated from (0..K-1)
);
```

## How the data is generated

1. Pick K random cluster centres in 9-D space, each coordinate in [-1000, 1000].
2. Pick a spread (std 1–10) per cluster.
3. For each row: choose a random cluster, generate an `ID`, and draw each feature from a Gaussian around that cluster's centre.
4. Round features to `DECIMAL(32,16)` and store the cluster as `true_label`.

## Using with K-means

Drop `ID` and `true_label` before clustering; use `true_label` only for scoring.

```python
import pandas as pd
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv("rows10000/synthetic_data_10000_rows_10_cols_5_clusters.csv")
X = df.drop(columns=["ID", "true_label"]).astype("float32").to_numpy()

# GPU (cuML, Linux/WSL2 + NVIDIA):
# import cupy as cp
# from cuml.cluster import KMeans
# km = KMeans(n_clusters=5, random_state=42).fit(cp.asarray(X))
# score = adjusted_rand_score(df["true_label"], km.labels_.get())
```

## Reproducibility

- `SEED = 42` gives identical output for the same arguments on any machine.
- Different K, rows, cols or seed produce different data.
- Pin library versions so others get the same data:
  ```bash
  pip freeze > requirements.txt
  ```
- Verify files match with `shasum rows10/*.csv`.

## Precision note

Values are generated as float64 (~15–16 significant digits), then stored as `DECIMAL(32,16)`; trailing decimal digits are zero-padded. Cast to `float32` for GPU K-means.
