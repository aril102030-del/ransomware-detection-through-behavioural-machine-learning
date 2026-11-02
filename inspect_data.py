import pandas as pd
import os

folder = r"C:\Users\aril1\.gemini\antigravity\scratch\ransomware_detection\dataset"
files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])

for f in files:
    path = os.path.join(folder, f)
    try:
        df = pd.read_csv(path)
        print(f"=== {f} ===")
        print(f"  Rows: {len(df)} | Cols: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")
        for col in df.select_dtypes(include=["object"]).columns[:5]:
            print(f"  [{col}] samples: {df[col].unique()[:5].tolist()}")
        print(f"  NaN counts: {df.isnull().sum()[df.isnull().sum()>0].to_dict()}")
        print()
    except Exception as e:
        print(f"  ERROR: {e}\n")
