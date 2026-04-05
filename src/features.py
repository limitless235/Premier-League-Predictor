import pandas as pd
import numpy as np
from pathlib import Path

def get_season(date):
    year = date.year
    month = date.month
    if month >= 8:
        start = year
        end = year + 1
    else:
        start = year - 1
        end = year
    return f"{str(start)[-2:]}{str(end)[-2:]}"

def build_features(df):
    """
    Build rolling features for each team based on their match history.
    
    Args:
        df: DataFrame with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG,
            home_xG, away_xG, home_elo_before, away_elo_before
    
    Returns:
        DataFrame with all original columns plus rolling features
    """
    # Work on a copy
    df = df.copy()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Date ascending
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates on Date, HomeTeam, AwayTeam keeping first
    df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
    
    # Filter out rows with NaN in team names
    df = df[df['HomeTeam'].notna() & df['AwayTeam'].notna()]
    
    # Get all unique teams
    teams = sorted(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist())
    teams = list(set(teams))
    
    # Build team history dictionaries
    team_history = {}
    for team in teams:
        # Get all matches involving this team
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        team_matches = team_matches.sort_values('Date').reset_index(drop=True)
        
        # Build per-match stats for each team
        team_matches['goals_scored'] = team_matches.apply(
            lambda row: row['FTHG'] if row['HomeTeam'] == team else row['FTAG'], axis=1
        )
        team_matches['goals_conceded'] = team_matches.apply(
            lambda row: row['FTAG'] if row['HomeTeam'] == team else row['FTHG'], axis=1
        )
        team_matches['xg_for'] = team_matches.apply(
            lambda row: row['home_xG'] if row['HomeTeam'] == team else row['away_xG'], axis=1
        )
        team_matches['xg_against'] = team_matches.apply(
            lambda row: row['away_xG'] if row['HomeTeam'] == team else row['home_xG'], axis=1
        )
        
        # Fill xG with goals if xG is null
        team_matches['xg_for'] = team_matches['xg_for'].fillna(team_matches['goals_scored'])
        team_matches['xg_against'] = team_matches['xg_against'].fillna(team_matches['goals_conceded'])
        
        # Points: 3 for win, 1 for draw, 0 for loss
        team_matches['points'] = team_matches.apply(
            lambda row: 3 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 0), axis=1
        )
        # Win: 1 for win, 0 otherwise
        team_matches['win'] = (team_matches['FTHG'] > team_matches['FTAG']).astype(int)
        # Clean sheet: 1 if conceded 0 else 0
        team_matches['clean_sheet'] = (team_matches['goals_conceded'] == 0).astype(int)
        
        # Store only the relevant columns for history
        team_history[team] = team_matches[['Date', 'goals_scored', 'goals_conceded', 
                                            'xg_for', 'xg_against', 'points', 'win', 
                                            'clean_sheet']].copy()
    
    # Derive Season from Date using the exact function provided
    df["Season"] = pd.to_datetime(df["Date"]).apply(get_season)
    
    # Create output DataFrame with all original columns
    output = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                 'home_xG', 'away_xG', 'home_elo_before', 'away_elo_before',
                 'Season']].copy()
    
    # Process each match
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        
        home_hist = team_history[home_team]
        away_hist = team_history[away_team]
        
        # Filter matches before current match
        home_mask = home_hist['Date'] < match_date
        away_mask = away_hist['Date'] < match_date
        
        home_past = home_hist[home_mask].sort_values('Date')
        away_past = away_hist[away_mask].sort_values('Date')
        
        # Calculate rolling stats for home team
        home_goals_5 = int(home_past['goals_scored'].tail(5).sum()) if len(home_past) >= 5 else int(home_past['goals_scored'].sum())
        home_goals_10 = int(home_past['goals_scored'].tail(10).sum()) if len(home_past) >= 10 else int(home_past['goals_scored'].sum())
        home_conceded_5 = int(home_past['goals_conceded'].tail(5).sum()) if len(home_past) >= 5 else int(home_past['goals_conceded'].sum())
        home_conceded_10 = int(home_past['goals_conceded'].tail(10).sum()) if len(home_past) >= 10 else int(home_past['goals_conceded'].sum())
        home_xg_5 = float(home_past['xg_for'].tail(5).sum()) if len(home_past) >= 5 else float(home_past['xg_for'].sum())
        home_xg_10 = float(home_past['xg_for'].tail(10).sum()) if len(home_past) >= 10 else float(home_past['xg_for'].sum())
        home_xg_against_5 = float(home_past['xg_against'].tail(5).sum()) if len(home_past) >= 5 else float(home_past['xg_against'].sum())
        home_xg_against_10 = float(home_past['xg_against'].tail(10).sum()) if len(home_past) >= 10 else float(home_past['xg_against'].sum())
        home_points_5 = int(home_past['points'].tail(5).sum()) if len(home_past) >= 5 else int(home_past['points'].sum())
        home_points_10 = int(home_past['points'].tail(10).sum()) if len(home_past) >= 10 else int(home_past['points'].sum())
        home_wins_5 = int(home_past['win'].tail(5).sum()) if len(home_past) >= 5 else int(home_past['win'].sum())
        home_wins_10 = int(home_past['win'].tail(10).sum()) if len(home_past) >= 10 else int(home_past['win'].sum())
        home_cs_5 = int(home_past['clean_sheet'].tail(5).sum()) if len(home_past) >= 5 else int(home_past['clean_sheet'].sum())
        home_cs_10 = int(home_past['clean_sheet'].tail(10).sum()) if len(home_past) >= 10 else int(home_past['clean_sheet'].sum())
        
        # Calculate rolling stats for away team
        away_goals_5 = int(away_past['goals_scored'].tail(5).sum()) if len(away_past) >= 5 else int(away_past['goals_scored'].sum())
        away_goals_10 = int(away_past['goals_scored'].tail(10).sum()) if len(away_past) >= 10 else int(away_past['goals_scored'].sum())
        away_conceded_5 = int(away_past['goals_conceded'].tail(5).sum()) if len(away_past) >= 5 else int(away_past['goals_conceded'].sum())
        away_conceded_10 = int(away_past['goals_conceded'].tail(10).sum()) if len(away_past) >= 10 else int(away_past['goals_conceded'].sum())
        away_xg_5 = float(away_past['xg_for'].tail(5).sum()) if len(away_past) >= 5 else float(away_past['xg_for'].sum())
        away_xg_10 = float(away_past['xg_for'].tail(10).sum()) if len(away_past) >= 10 else float(away_past['xg_for'].sum())
        away_xg_against_5 = float(away_past['xg_against'].tail(5).sum()) if len(away_past) >= 5 else float(away_past['xg_against'].sum())
        away_xg_against_10 = float(away_past['xg_against'].tail(10).sum()) if len(away_past) >= 10 else float(away_past['xg_against'].sum())
        away_points_5 = int(away_past['points'].tail(5).sum()) if len(away_past) >= 5 else int(away_past['points'].sum())
        away_points_10 = int(away_past['points'].tail(10).sum()) if len(away_past) >= 10 else int(away_past['points'].sum())
        away_wins_5 = int(away_past['win'].tail(5).sum()) if len(away_past) >= 5 else int(away_past['win'].sum())
        away_wins_10 = int(away_past['win'].tail(10).sum()) if len(away_past) >= 10 else int(away_past['win'].sum())
        away_cs_5 = int(away_past['clean_sheet'].tail(5).sum()) if len(away_past) >= 5 else int(away_past['clean_sheet'].sum())
        away_cs_10 = int(away_past['clean_sheet'].tail(10).sum()) if len(away_past) >= 10 else int(away_past['clean_sheet'].sum())
        
        # ELO difference
        home_elo = row['home_elo_before']
        away_elo = row['away_elo_before']
        if pd.isna(home_elo) or pd.isna(away_elo):
            home_elo = 1500
            away_elo = 1500
        
        elo_diff = home_elo - away_elo
        
        # Days since last match
        days_home = (match_date - home_past['Date'].max() if len(home_past) > 0 else pd.Timedelta(days=365)).days
        days_away = (match_date - away_past['Date'].max() if len(away_past) > 0 else pd.Timedelta(days=365)).days
        
        # Result encoding
        if row['FTHG'] > row['FTAG']:
            result = 2
        elif row['FTHG'] < row['FTAG']:
            result = 0
        else:
            result = 1
        
        # Add all rolling features to output
        output.loc[idx, 'home_goals_scored_last5'] = home_goals_5
        output.loc[idx, 'home_goals_scored_last10'] = home_goals_10
        output.loc[idx, 'home_goals_conceded_last5'] = home_conceded_5
        output.loc[idx, 'home_goals_conceded_last10'] = home_conceded_10
        output.loc[idx, 'home_xg_for_last5'] = home_xg_5
        output.loc[idx, 'home_xg_for_last10'] = home_xg_10
        output.loc[idx, 'home_xg_against_last5'] = home_xg_against_5
        output.loc[idx, 'home_xg_against_last10'] = home_xg_against_10
        output.loc[idx, 'home_points_last5'] = home_points_5
        output.loc[idx, 'home_points_last10'] = home_points_10
        output.loc[idx, 'home_wins_last5'] = home_wins_5
        output.loc[idx, 'home_wins_last10'] = home_wins_10
        output.loc[idx, 'home_clean_sheets_last5'] = home_cs_5
        output.loc[idx, 'home_clean_sheets_last10'] = home_cs_10
        
        output.loc[idx, 'away_goals_scored_last5'] = away_goals_5
        output.loc[idx, 'away_goals_scored_last10'] = away_goals_10
        output.loc[idx, 'away_goals_conceded_last5'] = away_conceded_5
        output.loc[idx, 'away_goals_conceded_last10'] = away_conceded_10
        output.loc[idx, 'away_xg_for_last5'] = away_xg_5
        output.loc[idx, 'away_xg_for_last10'] = away_xg_10
        output.loc[idx, 'away_xg_against_last5'] = away_xg_against_5
        output.loc[idx, 'away_xg_against_last10'] = away_xg_against_10
        output.loc[idx, 'away_points_last5'] = away_points_5
        output.loc[idx, 'away_points_last10'] = away_points_10
        output.loc[idx, 'away_wins_last5'] = away_wins_5
        output.loc[idx, 'away_wins_last10'] = away_wins_10
        output.loc[idx, 'away_clean_sheets_last5'] = away_cs_5
        output.loc[idx, 'away_clean_sheets_last10'] = away_cs_10
        
        output.loc[idx, 'elo_diff'] = elo_diff
        output.loc[idx, 'home_advantage'] = 1
        output.loc[idx, 'days_since_last_match_home'] = days_home
        output.loc[idx, 'days_since_last_match_away'] = days_away
        output.loc[idx, 'result'] = result
    
    return output

if __name__ == "__main__":
    input_path = Path('data/processed/matches.csv')
    output_path = Path('data/processed/matches_features.csv')
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Building features...")
    result = build_features(df)
    
    print(f"Saving to {output_path}...")
    result.to_csv(output_path, index=False)
    
    # Print only requested statistics
    print("\n" + "=" * 60)
    print("FEATURES BUILD COMPLETE")
    print("=" * 60)
    
    print(f"\nTotal rows: {len(result)}")
    
    print("\nUnique Season values and row counts:")
    season_counts = result['Season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count} rows")
    
    print("\nSample of 3 rows:")
    sample = result.head(3)
    for _, row in sample.iterrows():
        print(f"\n  Date: {row['Date'].date()}")
        print(f"  HomeTeam: {row['HomeTeam']}")
        print(f"  AwayTeam: {row['AwayTeam']}")
        print(f"  Season: {row['Season']}")