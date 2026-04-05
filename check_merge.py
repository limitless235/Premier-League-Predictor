import pandas as pd
from pathlib import Path
def check_merge():
    data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    name_map = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Newcastle": "Newcastle United",
        "Nott'm Forest": "Nottingham Forest",
        "QPR": "Queens Park Rangers",
        "West Brom": "West Bromwich Albion",
        "Wolves": "Wolverhampton Wanderers"
    }
    
    # Read all understat files and rename columns
    understat_files = sorted([f for f in data_dir.glob("understat_*.csv") if f.is_file()])
    understat_dfs = []
    
    for f in understat_files:
        df = pd.read_csv(f)
        df["Date"] = df["datetime"]
        df["HomeTeam"] = df["home"]
        df["AwayTeam"] = df["away"]
        understat_dfs.append(df)
    
    understat_df = pd.concat(understat_dfs, ignore_index=True)
    
    # Apply name mapping
    if "HomeTeam" in understat_df.columns and "AwayTeam" in understat_df.columns:
        understat_df["HomeTeam"] = understat_df["HomeTeam"].replace(name_map)
        understat_df["AwayTeam"] = understat_df["AwayTeam"].replace(name_map)
    
    # Convert Date to datetime and normalize
    understat_df["Date"] = pd.to_datetime(understat_df["Date"]).dt.normalize()
    
    # Read merged file
    merged_df = pd.read_csv(processed_dir / "matches.csv")
    merged_df["Date"] = pd.to_datetime(merged_df["Date"]).dt.normalize()
    
    # Anti-join using merge with indicator
    merged_cols = ["Date", "HomeTeam", "AwayTeam"]
    result = understat_df.merge(merged_df[merged_cols], on=merged_cols, how="left", indicator=True)
    unmatched = result[result["_merge"] == "left_only"][merged_cols]
    
    print(f"Total unmatched rows: {len(unmatched)}")
    if len(unmatched) > 0:
        print("\nUnmatched rows:")
        for _, row in unmatched.iterrows():
            print(f"  {row['Date']}, {row['HomeTeam']}, {row['AwayTeam']}")
    else:
        print("\nNo unmatched rows found")
if __name__ == "__main__":
    check_merge()