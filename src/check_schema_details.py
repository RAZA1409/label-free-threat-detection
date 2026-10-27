
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
dataset_dir = project_dir / "dataset"

for file_path in sorted(dataset_dir.rglob("*.csv")):
    print("\n" + "=" * 70)
    print(file_path.name)

    df = pd.read_csv(file_path, nrows=2, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"Total columns: {len(df.columns)}")
    print("\nLast 20 columns:")
    for column in df.columns[-20:]:
        print(repr(column))

    print("\nPotential identifying/time columns:")
    for column in df.columns:
        if any(word in column.lower() for word in
               ["time", "date", "flow id", "source ip", "destination ip"]):
            print(repr(column))

    print("\nFirst row label:", repr(df["Label"].iloc[0]))
