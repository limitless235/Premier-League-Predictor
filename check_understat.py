import pandas as pd
from pathlib import Path
def check_understat():
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print("Error: data/raw/ directory not found")
        return
    
    csv_files = sorted([f for f in data_dir.glob("understat_*.csv") if f.is_file()])
    
    if not csv_files:
        print("No understat_*.csv files found in data/raw/")
        return
    
    all_dfs = []
    all_teams = set()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        season = csv_file.stem[8:]
        
        print(f"\n=== {csv_file.name} ===")
        print(f"Rows: {len(df)}")
        
        if "datetime" in df.columns and len(df) > 0:
            df_dates = pd.to_datetime(df["datetime"])
            print(f"Date range: {df_dates.min()} to {df_dates.max()}")
        else:
            print("Date range: N/A")
        
        bad_rows = df[df["data_quality_flag"] == False]
        if len(bad_rows) > 0:
            print(f"Bad rows (data_quality_flag=False): {len(bad_rows)}")
        else:
            print("Bad rows: None")
        
        if "home_xG" in df.columns and "away_xG" in df.columns and len(df) > 0:
            print(f"home_xG: {df['home_xG'].min():.2f} to {df['home_xG'].max():.2f}")
            print(f"away_xG: {df['away_xG'].min():.2f} to {df['away_xG'].max():.2f}")
        
        if "home" in df.columns and "away" in df.columns and len(df) > 0:
            all_teams.update(df["home"].dropna().unique())
            all_teams.update(df["away"].dropna().unique())
        
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print("\n=== COMBINED DATA ===")
    print(f"Total matches: {len(combined)}")
    print(f"Unique teams: {sorted(all_teams)}")
if __name__ == "__main__":
    check_understat()