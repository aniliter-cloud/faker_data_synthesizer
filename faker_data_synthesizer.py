from faker import Faker
from faker.providers import BaseProvider
import numpy as np
import pandas as pd
from decimal import Decimal




COLS = 10

CLUSTER_PLAN = {
    10_000:  [3, 5, 7],
    50_000:  [7, 9, 11],
    100_000: [13, 15, 17],
}

MAX_ROWS   = list(CLUSTER_PLAN)

N_CLUSTERS = 5
SEED       = 42
SCALE      = Decimal("1." + "0" * 16)

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

    Parameters:
    - max_rows (int): The maximum number of rows in the dataset.
    - cols (int): The number of columns in the dataset.
    - n_clusters (int): The number of clusters to generate. Default is N_CLUSTERS.
    - seed (int | None): The seed for random number generation. Default is SEED.

    Returns:
    - pd.DataFrame: A DataFrame containing the synthetic dataset.
    """

    faker = make_faker(seed)
    rnd = faker.random
    col_names = ["ID"] + [f"col_{i}" for i in range(1, cols)]
    n_feat = cols - 1

    centers = [[rnd.uniform(-100, 1000) for _ in range(n_feat)] for _ in range(n_clusters)]
    std = [rnd.uniform(1, 10) for _ in range(n_clusters)]

    rows, labels = [], []
    for i in range(max_rows):
        cluster = rnd.randint(0, n_clusters - 1)
        labels.append(cluster)                      # int label: 0..K-1

        row = [faker.col_id()]
        for j in range(n_feat):
            value = rnd.gauss(centers[cluster][j], std[cluster])
            row.append(Decimal(repr(value)).quantize(SCALE))
        rows.append(row)

    return pd.DataFrame(rows, columns=col_names), np.array(labels)


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


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, help="rows for a single test run")
    parser.add_argument("--cols", type=int, default=COLS, help="total cols incl. ID")
    parser.add_argument("--clusters", type=int, nargs="+", default=[3],
                        help="one or more K values, e.g. --clusters 3 5 7")
    args = parser.parse_args()

    print(ddl("synthetic_data", args.cols))

    # custom run: python data_synthesizer.py --rows 10 --cols 10 --clusters 3 5 7
    if args.rows:
        for k in args.clusters:
            df, labels = synthesizer(args.rows, args.cols, n_clusters=k)
            print(df)
            print(labels)
            folder = f"rows{args.rows}"
            os.makedirs(folder, exist_ok=True)      # create if not present
            name = os.path.join(folder, f"synthetic_data_{args.rows}_rows_{args.cols}_cols_{k}_clusters.csv")
            df["true_label"] = labels               # answer key; drop before K-means
            df.to_csv(name, index=False)
            print(f"Saved synthetic dataset to {name}.")
        raise SystemExit

    # full plan (no --rows): 9 datasets
    for max_rows, clusters in CLUSTER_PLAN.items():
        for k in clusters:
            df, labels = synthesizer(max_rows, COLS, n_clusters=k)
            print(f"Generated DataFrame with {max_rows} rows and {COLS} columns for {k} clusters.")
            print(df.head())
            folder = f"rows{max_rows}"
            os.makedirs(folder, exist_ok=True)      # create if not present
            name = os.path.join(folder, f"synthetic_data_{max_rows}_rows_{COLS}_cols_{k}_clusters.csv")
            df.to_csv(name, index=False)
            print(f"Saved synthetic dataset to {name}.")

    faker1, faker2 = make_faker(seed=SEED), make_faker(seed=SEED)
    assert faker1.col_id() == faker2.col_id(), "Faker instances with the same seed should produce the same column ID."