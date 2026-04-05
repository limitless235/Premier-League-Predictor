import pandas as pd
from pathlib import Path

def check_data():
    data_dir = Path("data/raw")
    required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", 
                       "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A"]
    
    if not data_dir.exists():
        print("Error: data/raw/ directory not found")
        return
    
    csv_files = sorted([f for f in data_dir.glob("fdco_*.csv") if f.is_file()])
    
    if not csv_files:
        print("No fdco_*.csv files found in data/raw/")
        return
    
    all_dfs = []
    all_teams = set()
    seasons_with_odds = 0
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        print(f"\n=== {csv_file.name} ===")
        print(f"Rows: {len(df)}")
        
        if "Date" in df.columns and len(df) > 0:
            df_dates = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
            print(f"Date range: {df_dates.min()} to {df_dates.max()}")
        elif "Date" in df.columns:
            print("Date range: N/A (no data rows)")
        else:
            print("Date range: N/A (column missing)")
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"Missing columns: {missing}")
        else:
            print("Missing columns: None")
        
        all_dfs.append(df)
        
        if "HomeTeam" in df.columns and "AwayTeam" in df.columns and len(df) > 0:
            all_teams.update(df["HomeTeam"].dropna().unique())
            all_teams.update(df["AwayTeam"].dropna().unique())
        
        if "B365H" in df.columns and len(df) > 0:
            has_odds = df[["B365H", "B365D", "B365A"]].notna().all().all()
            if has_odds:
                seasons_with_odds += 1
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print("\n=== COMBINED DATA ===")
    print(f"Total matches: {len(combined)}")
    print(f"Unique teams: {sorted(all_teams)}")
    print(f"Seasons with B365 odds: {seasons_with_odds}/{len(csv_files)}")

if __name__ == "__main__":
    check_data()