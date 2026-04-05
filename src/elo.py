import pandas as pd
import numpy as np
import math
from pathlib import Path

class EloSystem:
    def __init__(self, k=32, regression=0.3):
        self.k = k
        self.regression = regression
        self.ratings = {}
    
    def get_rating(self, team):
        if team not in self.ratings:
            self.ratings[team] = 1500.0
        return self.ratings[team]
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def margin_multiplier(self, goal_diff):
        multiplier = math.log(abs(goal_diff) + 1) * 2.5
        return max(1.0, min(2.5, multiplier))
    
    def update(self, home_team, away_team, home_goals, away_goals):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        actual_home = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        actual_away = 1.0 if away_goals > home_goals else (0.5 if away_goals == home_goals else 0.0)
        
        expected_home = self.expected_score(home_rating, away_rating)
        expected_away = self.expected_score(away_rating, home_rating)
        
        goal_diff = abs(home_goals - away_goals)
        multiplier = self.margin_multiplier(goal_diff)
        
        new_home_rating = home_rating + (self.k * multiplier * (actual_home - expected_home))
        new_away_rating = away_rating + (self.k * multiplier * (actual_away - expected_away))
        
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
        
        return (new_home_rating, new_away_rating)
    
    def season_reset(self, teams_in_season):
        for team in teams_in_season:
            if team in self.ratings:
                old_rating = self.ratings[team]
                self.ratings[team] = old_rating + (1500 - old_rating) * self.regression
            else:
                self.ratings[team] = 1500.0
    
    def process_matches(self, matches_df):
        matches_df = matches_df.copy()
        matches_df['home_elo_before'] = 0.0
        matches_df['away_elo_before'] = 0.0
        matches_df['elo_diff'] = 0.0
        
        current_season_teams = []
        current_season = None
        
        for idx, row in matches_df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            home_goals = row['FTHG']
            away_goals = row['FTAG']
            date = row['Date']
            
            season_key = date.year
            if current_season != season_key:
                current_season = season_key
                all_teams_in_season = current_season_teams + [home_team, away_team]
                self.season_reset(all_teams_in_season)
                current_season_teams = [home_team, away_team]
            else:
                current_season_teams = list(set(current_season_teams + [home_team, away_team]))
            
            home_elo = self.get_rating(home_team)
            away_elo = self.get_rating(away_team)
            
            matches_df.loc[idx, 'home_elo_before'] = home_elo
            matches_df.loc[idx, 'away_elo_before'] = away_elo
            matches_df.loc[idx, 'elo_diff'] = home_elo - away_elo
            
            self.update(home_team, away_team, home_goals, away_goals)
        
        return matches_df

if __name__ == "__main__":
    data_dir = Path("data/processed")
    matches_df = pd.read_csv(data_dir / "matches.csv")
    
    matches_df['Date'] = pd.to_datetime(matches_df['Date'])
    matches_df = matches_df.sort_values('Date')
    
    elo_system = EloSystem()
    matches_df = elo_system.process_matches(matches_df)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(data_dir / "matches.csv", index=False)
    
    all_ratings = []
    for idx, row in matches_df.iterrows():
        all_ratings.extend([row['home_elo_before'], row['away_elo_before']])
    
    print(f"Min Elo: {min(all_ratings):.2f}")
    print(f"Max Elo: {max(all_ratings):.2f}")
    print(f"Mean Elo: {np.mean(all_ratings):.2f}")
    
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
    man_city = mc_df.groupby(mc_df['Date'].dt.year)['home_elo_before'].mean()
    for season, elo in man_city.sort_index().items():
        print(f"  {season}: {elo:.2f}")