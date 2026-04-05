import pandas as pd

def check_model_data():
    df = pd.read_csv("data/processed/matches_features.csv")
    
    print("=" * 60)
    print("MATCHES FEATURES DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal rows: {len(df)}")
    
    print("\nSeason Distribution:")
    print("-" * 40)
    if 'Season' in df.columns:
        season_counts = df['Season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"  {season}: {count} matches")
    
    print("\nDuplicate Rows Analysis:")
    print("-" * 40)
    if all(col in df.columns for col in ['Date', 'HomeTeam', 'AwayTeam']):
        duplicates = df.duplicated(subset=['Date', 'HomeTeam', 'AwayTeam'], keep=False)
        dup_count = duplicates.sum()
        print(f"  Total duplicate rows (Date, HomeTeam, AwayTeam): {dup_count}")
        if dup_count > 0:
            print(f"  Unique combinations: {df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam']).shape[0]}")
    
    print("\nOverall Average Goals:")
    print("-" * 40)
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        avg_fthg = df['FTHG'].mean()
        avg_ftag = df['FTAG'].mean()
        avg_total = (avg_fthg + avg_ftag) / 2
        print(f"  Average Home Goals (FTHG): {avg_fthg:.2f}")
        print(f"  Average Away Goals (FTAG): {avg_ftag:.2f}")
        print(f"  Average Total Goals per Game: {avg_total:.2f}")
    
    print("\nTop 6 Teams Average Goals:")
    print("-" * 40)
    top_teams = ['Liverpool', 'Manchester City', 'Arsenal', 'Chelsea', 'Tottenham', 'Manchester United']
    
    if all(col in df.columns for col in ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']):
        for team in top_teams:
            team_home = df[df['HomeTeam'] == team]['FTHG'].mean()
            team_away = df[df['AwayTeam'] == team]['FTAG'].mean()
            team_avg = (team_home + team_away) / 2
            print(f"  {team}:")
            print(f"    Home Goals: {team_home:.2f}")
            print(f"    Away Goals: {team_away:.2f}")
            print(f"    Average: {team_avg:.2f}")
    
    print("\nData Types:")
    print("-" * 40)
    if 'Season' in df.columns:
        print(f"  Season column dtype: {df['Season'].dtype}")
    
    print("\n" + "=" * 60)
    print("ADDITIONAL STATISTICS")
    print("=" * 60)
    
    print(f"\nColumns in dataframe: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    print("\nNull Values per Column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"  {col}: {count}")
    
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        print(f"\nNull FTHG values: {df['FTHG'].isnull().sum()}")
        print(f"Null FTAG values: {df['FTAG'].isnull().sum()}")
    
    if 'Date' in df.columns:
        print(f"\nDate Range:")
        print(f"  Min: {df['Date'].min()}")
        print(f"  Max: {df['Date'].max()}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    check_model_data()