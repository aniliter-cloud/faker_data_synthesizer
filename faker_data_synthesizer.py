from faker import Faker
from faker.providers import BaseProvider
import numpy as np
import pandas as pd
import time




COLS = 10

CLUSTER_PLAN = {
    1_000_000:  [3, 5, 7],
    5_000_000:  [7, 9, 11],
    10_000_000: [13, 15, 17],
}

MAX_ROWS   = list(CLUSTER_PLAN)

N_CLUSTERS = 5
SEED       = 42
DECIMALS   = 16                     # DECIMAL(32,16) -> 16 decimal places in CSV

class ColIdProvider(BaseProvider):
    """
    A custom provider for generating column identifiers.
    """
    def col_id(self) -> str:
        """
        Generates a random column identifier.

        Returns:
        - str: A random column identifier.
        """
        return self.generator.uuid4().replace("-", "")


def make_faker(seed: int | None = SEED) -> Faker:
    """
    Creates a Faker instance with an optional seed for reproducibility.

    Parameters:
    - seed (int | None): The seed for random number generation. Default is SEED.

    Returns:
    - Faker: An instance of the Faker class.
    """
    faker = Faker()
    faker.add_provider(ColIdProvider)
    if seed is not None:
        faker.seed_instance(seed)
    return faker


def synthesizer(max_rows: int, cols: int, n_clusters:int = N_CLUSTERS,seed:int | None =SEED):
    """
    Generates a synthetic dataset with specified number of rows, columns, and clusters.
    Features are generated with vectorised NumPy (fast for millions of rows);
    IDs still come from faker.col_id().

    Parameters:
    - max_rows (int): The maximum number of rows in the dataset.
    - cols (int): The number of columns in the dataset.
    - n_clusters (int): The number of clusters to generate. Default is N_CLUSTERS.
    - seed (int | None): The seed for random number generation. Default is SEED.

    Returns:
    - pd.DataFrame: A DataFrame containing the synthetic dataset.
    - np.ndarray: True cluster label per row (0..K-1).
    """

    faker = make_faker(seed)
    rng = np.random.default_rng(seed)
    col_names = ["ID"] + [f"col_{i}" for i in range(1, cols)]
    n_feat = cols - 1

    centers = rng.uniform(-100, 1000, size=(n_clusters, n_feat))
    std = rng.uniform(1, 10, size=n_clusters)

    # one cluster per row, then all values at once: centre + noise * std
    labels = rng.integers(0, n_clusters, size=max_rows)
    values = centers[labels] + rng.standard_normal((max_rows, n_feat)) * std[labels, None]

    df = pd.DataFrame(values, columns=col_names[1:])
    df.insert(0, "ID", [faker.col_id() for _ in range(max_rows)])

    return df, labels


def ddl(table_name: str, cols: int) -> str:
    """
    Generates a SQL DDL statement for creating a table with specified columns.

    Parameters:
    - table_name (str): The name of the table.
    - cols (int): The number of columns in the table.

    Returns:
    - str: A SQL DDL statement for creating the table.
    """
    col_names = [f"col_{i}" for i in range(1, cols)]
    col_defs = ",\n  ".join(["ID CHAR(32)"] + [f"{name} DECIMAL(32, 16)" for name in col_names] + ["true_label INT"])
    ddl_statement = f"CREATE TABLE {table_name} (\n  {col_defs}\n);"
    return ddl_statement


def save(df: pd.DataFrame, labels: np.ndarray, max_rows: int, cols: int, k: int) -> str:
    """Adds true_label and writes the CSV into rows<N>/ (created if missing)."""
    import os
    folder = f"rows{max_rows}"
    os.makedirs(folder, exist_ok=True)
    name = os.path.join(folder, f"synthetic_data_{max_rows}_rows_{cols}_cols_{k}_clusters.csv")
    df["true_label"] = labels               # answer key; drop before K-means
    df.to_csv(name, index=False, float_format=f"%.{DECIMALS}f")
    return name


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, help="rows for a single test run")
    parser.add_argument("--cols", type=int, default=COLS, help="total cols incl. ID")
    parser.add_argument("--clusters", type=int, nargs="+", default=[3],
                        help="one or more K values, e.g. --clusters 3 5 7")
    args = parser.parse_args()

    print(ddl("synthetic_data", args.cols))

    # custom run: python faker_data_synthesizer.py --rows 10 --cols 10 --clusters 3 5 7
    if args.rows:
        plan, cols = {args.rows: args.clusters}, args.cols
    else:
        plan, cols = CLUSTER_PLAN, COLS      # full plan: 1M / 5M / 10M

    for max_rows, clusters in plan.items():
        for k in clusters:
            t0 = time.perf_counter()
            df, labels = synthesizer(max_rows, cols, n_clusters=k)
            t1 = time.perf_counter()
            print(df.head())
            name = save(df, labels, max_rows, cols, k)
            t2 = time.perf_counter()
            print(f"Saved {name}  (generate {t1 - t0:.1f}s, write {t2 - t1:.1f}s)")

    faker1, faker2 = make_faker(seed=SEED), make_faker(seed=SEED)
    assert faker1.col_id() == faker2.col_id(), "Faker instances with the same seed should produce the same column ID."