import pandas as pd
from pathlib import Path

def fix_merge():
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
    
    fdco_files = sorted([f for f in data_dir.glob("fdco_*.csv") if f.is_file()])
    fdco_dfs = [pd.read_csv(f) for f in fdco_files]
    fdco_df = pd.concat(fdco_dfs, ignore_index=True)
    
    if "HomeTeam" in fdco_df.columns and "AwayTeam" in fdco_df.columns:
        fdco_df["HomeTeam"] = fdco_df["HomeTeam"].replace(name_map)
        fdco_df["AwayTeam"] = fdco_df["AwayTeam"].replace(name_map)
    
    if "Date" in fdco_df.columns:
        fdco_df["Date"] = pd.to_datetime(fdco_df["Date"], format="mixed", dayfirst=True).dt.normalize()
    
    understat_files = sorted([f for f in data_dir.glob("understat_*.csv") if f.is_file()])
    understat_dfs = []
    
    for f in understat_files:
        df = pd.read_csv(f)
        df["Date"] = df["datetime"]
        df["HomeTeam"] = df["home"]
        df["AwayTeam"] = df["away"]
        understat_dfs.append(df)
    
    understat_df = pd.concat(understat_dfs, ignore_index=True)
    understat_df["Date"] = pd.to_datetime(understat_df["Date"]).dt.normalize()
    
    understat_df["season"] = understat_df["Date"].dt.year
    
    merge_cols = ["Date", "HomeTeam", "AwayTeam"]
    result = fdco_df.merge(understat_df, on=merge_cols, how="left")
    
    result["has_xg"] = result["home_xG"].notna()
    
    rows_fixed = 0
    
    for day_offset in [-1, 1]:
        shifted_df = understat_df.copy()
        shifted_df["Date"] = shifted_df["Date"] + pd.Timedelta(days=day_offset)
        
        fuzzy_result = fdco_df.merge(shifted_df, on=merge_cols, how="inner")
        
        for _, row in fuzzy_result.iterrows():
            mask = (
                (result["Date"] == row["Date"]) & 
                (result["HomeTeam"] == row["HomeTeam"]) & 
                (result["AwayTeam"] == row["AwayTeam"]) & 
                (~result["has_xg"])
            )
            
            if mask.any():
                result.loc[mask, "home_xG"] = row["home_xG"]
                result.loc[mask, "away_xG"] = row["away_xG"]
                result.loc[mask, "has_xg"] = True
                rows_fixed += mask.sum()
    
    final_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", 
                  "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A", "home_xG", "away_xG", "has_xg"]
    
    available_cols = [c for c in final_cols if c in result.columns]
    result = result[available_cols]
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(processed_dir / "matches.csv", index=False)
    
    print(f"Total rows: {len(result)}")
    print(f"Rows with xG: {result['has_xg'].sum()}")
    print(f"Rows without xG: {(~result['has_xg']).sum()}")
    print(f"Rows fixed by fuzzy date match: {rows_fixed}")

if __name__ == "__main__":
    fix_merge()