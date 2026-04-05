import pandas as pd

def analyze_elos():
    matches_df = pd.read_csv("data/processed/matches.csv")
    
    matches_df['Date'] = pd.to_datetime(matches_df['Date'])
    
    print("\nTop 10 highest home_elo_before:")
    top_high = matches_df.nlargest(10, 'home_elo_before')[['Date', 'HomeTeam', 'AwayTeam', 'home_elo_before']]
    for _, row in top_high.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}, {row['HomeTeam']}, {row['AwayTeam']}, {row['home_elo_before']:.2f}")
    
    print("\nTop 10 lowest home_elo_before:")
    top_low = matches_df.nsmallest(10, 'home_elo_before')[['Date', 'HomeTeam', 'AwayTeam', 'home_elo_before']]
    for _, row in top_low.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}, {row['HomeTeam']}, {row['AwayTeam']}, {row['home_elo_before']:.2f}")
    
    print("\nAverage Elo per team (descending):")
    team_avg = matches_df.groupby('HomeTeam')['home_elo_before'].mean().sort_values(ascending=False)
    print(team_avg.round(2))
    
    print("\nManchester City mean home_elo_before by season:")
    mc_df = matches_df[matches_df['HomeTeam'] == 'Manchester City']
    man_city_seasons = mc_df.groupby(mc_df['Date'].dt.year)['home_elo_before'].mean()
    
    for season, elo in man_city_seasons.sort_index().items():
        print(f"  {season}: {elo:.2f}")

if __name__ == "__main__":
    analyze_elos()