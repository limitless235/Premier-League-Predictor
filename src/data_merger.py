import pandas as pd
from pathlib import Path

def merge_datasets():
    data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    name_map = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Newcastle": "Newcastle United",
        "Nott'm Forest": "Nottingham Forest",
        "QPR": "Queens Park Rangers",
        "West Brom": "West Bromwich Albion",
        "Wolves": "Wolverhampton Wanderers"
    }
    
    fdco_files = sorted([f for f in data_dir.glob("fdco_*.csv") if f.is_file()])
    understat_files = sorted([f for f in data_dir.glob("understat_*.csv") if f.is_file()])
    
    if not fdco_files:
        print("No fdco_*.csv files found")
        return
    if not understat_files:
        print("No understat_*.csv files found")
        return
    
    fdco_dfs = []
    for f in fdco_files:
        df = pd.read_csv(f)
        if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
            df[["HomeTeam", "AwayTeam"]] = df[["HomeTeam", "AwayTeam"]].replace(name_map)
        fdco_dfs.append(df)
    
    fdco_df = pd.concat(fdco_dfs, ignore_index=True)
    fdco_df["Date"] = pd.to_datetime(fdco_df["Date"], format="mixed", dayfirst=True)
    
    understat_dfs = []
    for f in understat_files:
        df = pd.read_csv(f)
        understat_dfs.append(df)
    
    understat_df = pd.concat(understat_dfs, ignore_index=True)
    understat_df["Date"] = pd.to_datetime(understat_df["datetime"]).dt.normalize()
    understat_df = understat_df.rename(columns={"home": "HomeTeam", "away": "AwayTeam"})
    
    merged = pd.merge(fdco_df, understat_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    merged["has_xg"] = merged[["home_xG", "away_xG"]].notna().all(axis=1)
    
    duplicates = merged.duplicated(subset=["Date", "HomeTeam", "AwayTeam"])
    
    merged.to_csv("data/processed/matches.csv", index=False)
    
    print(f"Total rows: {len(merged)}")
    print(f"Rows with xG: {merged['has_xg'].sum()}")
    print(f"Rows without xG: {len(merged) - merged['has_xg'].sum()}")
    if duplicates.sum() > 0:
        print(f"Duplicate rows: {duplicates.sum()}")
    else:
        print("Duplicate rows: 0")

if __name__ == "__main__":
    merge_datasets()