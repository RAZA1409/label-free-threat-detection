
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "GeneratedLabelledFlows"

path = next(DATA_DIR.rglob(
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
))

df = pd.read_csv(
    path,
    encoding="cp1252",
    low_memory=False
)
df.columns = df.columns.str.strip()

print("Total rows:", len(df))

# Rows where both timestamp and label are missing
bad = df["Timestamp"].isna() & df["Label"].isna()
print("Rows missing both timestamp and label:", int(bad.sum()))

# Inspect the first and last affected rows
positions = df.index[bad]

if len(positions):
    first = int(positions[0])
    last = int(positions[-1])

    print("\nFirst affected row index:", first)
    print(df.iloc[max(0, first - 2):first + 3, :8].to_string())

    print("\nLast affected row index:", last)
    print(df.iloc[max(0, last - 2):last + 3, :8].to_string())

# Check whether affected rows are entirely empty
print("\nMissing cells among affected rows:",
      int(df.loc[bad].isna().sum().sum()))
print("Total cells among affected rows:",
      int(df.loc[bad].size))

print("\nMissingness by column, top 15:")
print(df.loc[bad].isna().sum().sort_values(ascending=False).head(15))

del df
