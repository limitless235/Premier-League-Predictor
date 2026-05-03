import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.model import DixonColesModel

def build_current_season_table(results_df):
    def get_team_stats(home_team, away_team, home_goals, away_goals):
        home_points = 3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
        away_points = 3 if away_goals > home_goals else (1 if away_goals == home_goals else 0)
        
        home_stats = {
            'Team': home_team,
            'Played': 1,
            'Won': 1 if home_goals > away_goals else 0,
            'Drawn': 1 if home_goals == away_goals else 0,
            'Lost': 1 if home_goals < away_goals else 0,
            'GF': home_goals,
            'GA': away_goals,
            'Points': home_points
        }
        
        away_stats = {
            'Team': away_team,
            'Played': 1,
            'Won': 1 if away_goals > home_goals else 0,
            'Drawn': 1 if away_goals == home_goals else 0,
            'Lost': 1 if away_goals < home_goals else 0,
            'GF': away_goals,
            'GA': home_goals,
            'Points': away_points
        }
        
        return home_stats, away_stats
    
    stats_list = []
    for _, row in results_df.iterrows():
        home_stats, away_stats = get_team_stats(
            row['HomeTeam'], 
            row['AwayTeam'], 
            row['FTHG'], 
            row['FTAG']
        )
        stats_list.append(home_stats)
        stats_list.append(away_stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    table = stats_df.groupby('Team').agg({
        'Played': 'sum',
        'Won': 'sum',
        'Drawn': 'sum',
        'Lost': 'sum',
        'GF': 'sum',
        'GA': 'sum',
        'Points': 'sum'
    }).reset_index()
    
    table['GD'] = table['GF'] - table['GA']
    
    table = table.sort_values(
        by=['Points', 'GD', 'GF'],
        ascending=[False, False, False]
    )
    
    return table

def simulate_season(model, fixtures_df, current_table, n_simulations=10000):
    n_teams = len(current_table)
    champion_counts = np.zeros(n_teams)
    top4_counts = np.zeros(n_teams)
    relegated_counts = np.zeros(n_teams)
    total_points = np.zeros(n_teams)
    team_points = np.zeros((n_simulations, n_teams))
    team_positions = np.zeros((n_simulations, n_teams), dtype=int)
    
    teams = current_table['Team'].tolist()
    points_lookup = dict(zip(current_table['Team'], current_table['Points']))
    team_idx = {team: i for i, team in enumerate(teams)}
    fixtures = fixtures_df[['HomeTeam', 'AwayTeam']].values.tolist()
    
    predictions = []
    for home_team, away_team in fixtures:
        pred = model.predict_match(home_team, away_team)
        predictions.append(pred)
    
    for sim in range(n_simulations):
        standings = current_table.copy()
        
        for i, (home_team, away_team) in enumerate(fixtures):
            pred = predictions[i]
            prob_home = pred['home_win']
            prob_draw = pred['draw']
            prob_away = pred['away_win']
            
            r = np.random.random()
            if r < prob_home:
                home_pts, away_pts = 3, 0
            elif r < prob_home + prob_draw:
                home_pts, away_pts = 1, 1
            else:
                home_pts, away_pts = 0, 3
            
            home_idx = team_idx[home_team]
            away_idx = team_idx[away_team]
            
            if home_pts == 3:
                standings.iloc[home_idx, standings.columns.get_loc('Won')] += 1
                standings.iloc[home_idx, standings.columns.get_loc('Points')] += 3
                standings.iloc[away_idx, standings.columns.get_loc('Lost')] += 1
                standings.iloc[away_idx, standings.columns.get_loc('Points')] += 0
            elif home_pts == 1:
                standings.iloc[home_idx, standings.columns.get_loc('Drawn')] += 1
                standings.iloc[home_idx, standings.columns.get_loc('Points')] += 1
                standings.iloc[away_idx, standings.columns.get_loc('Drawn')] += 1
                standings.iloc[away_idx, standings.columns.get_loc('Points')] += 1
            else:
                standings.iloc[home_idx, standings.columns.get_loc('Lost')] += 1
                standings.iloc[home_idx, standings.columns.get_loc('Points')] += 0
                standings.iloc[away_idx, standings.columns.get_loc('Won')] += 1
                standings.iloc[away_idx, standings.columns.get_loc('Points')] += 3
            
            team_points[sim, home_idx] = standings.iloc[home_idx, standings.columns.get_loc('Points')]
            team_points[sim, away_idx] = standings.iloc[away_idx, standings.columns.get_loc('Points')]
        
        standings = standings.sort_values(
            by=['Points', 'GD', 'GF'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        for rank in range(n_teams):
            team_name = standings.iloc[rank]['Team']
            t_idx = team_idx[team_name]
            team_positions[sim, t_idx] = rank + 1
            
            if rank == 0:
                champion_counts[t_idx] += 1
            if rank < 4:
                top4_counts[t_idx] += 1
            if rank >= n_teams - 3:
                relegated_counts[t_idx] += 1
    
    champion_prob = {team: counts / n_simulations for team, counts in zip(teams, champion_counts)}
    top4_prob = {team: counts / n_simulations for team, counts in zip(teams, top4_counts)}
    relegated_prob = {team: counts / n_simulations for team, counts in zip(teams, relegated_counts)}
    expected_points = {team: np.mean(team_points[:, team_idx[team]]) for team in teams}
    points_std = {team: np.std(team_points[:, team_idx[team]]) for team in teams}
    
    return {
        'champion_prob': champion_prob,
        'top4_prob': top4_prob,
        'relegated_prob': relegated_prob,
        'expected_points': expected_points,
        'points_std': points_std,
        'team_positions': team_positions,
        'team_idx': team_idx,
        'teams': teams
    }

def print_simulation_results(results, n_simulations):
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    
    print("\n TITLE RACE (Champion Probability > 1%)")
    print("-" * 80)
    title_teams = [(team, prob) for team, prob in results['champion_prob'].items() if prob > 0.01]
    title_teams.sort(key=lambda x: x[1], reverse=True)
    
    if title_teams:
        for team, prob in title_teams:
            exp_pts = results['expected_points'][team]
            std_pts = results['points_std'][team]
            print(f"{team:20} {prob*100:6.2f}%  (Exp: {exp_pts:6.1f} pts, Std: {std_pts:.1f})")
    else:
        print("  No team has > 1% chance of winning the title")
    
    print("\n TOP 4 RACE (Top 4 Probability > 5%)")
    print("-" * 80)
    top4_teams = [(team, prob) for team, prob in results['top4_prob'].items() if prob > 0.05]
    top4_teams.sort(key=lambda x: x[1], reverse=True)
    
    if top4_teams:
        for team, prob in top4_teams:
            exp_pts = results['expected_points'][team]
            std_pts = results['points_std'][team]
            print(f"  {team:20} {prob*100:6.2f}%  (Exp: {exp_pts:6.1f} pts, Std: {std_pts:.1f})")
    else:
        print("  No team has > 5% chance of top 4 finish")
    
    print("\n  RELEGATION BATTLE (Relegation Probability > 1%)")
    print("-" * 80)
    relegation_teams = [(team, prob) for team, prob in results['relegated_prob'].items() if prob > 0.01]
    relegation_teams.sort(key=lambda x: x[1], reverse=True)
    
    if relegation_teams:
        for team, prob in relegation_teams:
            exp_pts = results['expected_points'][team]
            std_pts = results['points_std'][team]
            print(f"  {team:20} {prob*100:6.2f}%  (Exp: {exp_pts:6.1f} pts, Std: {std_pts:.1f})")
    else:
        print("  No team has > 1% chance of relegation")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Simulations: {n_simulations:,}")
    print(f"  Expected Champion: {max(results['champion_prob'], key=results['champion_prob'].get)}")
    print(f"  Expected Top 4: {max(results['top4_prob'], key=results['top4_prob'].get)}")
    print(f"  Most Likely to Relegate: {max(results['relegated_prob'], key=results['relegated_prob'].get)}")

def print_relegation_details(results, n_simulations=10000):
    print("\n" + "=" * 80)
    print("DETAILED RELEGATION ANALYSIS")
    print("=" * 80)
    
    print("\n RELEGATION PROBABILITY (Top 3 Most Likely)")
    print("-" * 80)
    
    relegation_teams = [(team, prob) for team, prob in results['relegated_prob'].items() if prob > 0.001]
    relegation_teams.sort(key=lambda x: x[1], reverse=True)
    
    if relegation_teams:
        for team, prob in relegation_teams[:10]:
            exp_pts = results['expected_points'][team]
            std_pts = results['points_std'][team]
            print(f"  {team:20} {prob*100:6.2f}%  (Exp: {exp_pts:6.1f} pts, Std: {std_pts:.1f})")
    
    print("\n RELEGATION COUNTS (How many times each team was relegated)")
    print("-" * 80)
    print(f"{'Team':<20} {'Relegated':>10} {'Relegation %':>12}")
    print("-" * 80)
    
    for team in results['relegated_prob'].keys():
        prob = results['relegated_prob'][team]
        count = int(prob * n_simulations)
        exp_pts = results['expected_points'][team]
        std_pts = results['points_std'][team]
        
        print(f"{team:<20} {count:>10}  {prob*100:>11.2f}%  (Exp: {exp_pts:.1f} pts)")
    
    print("\n TEAMS AT RISK (Relegation Probability > 5%)")
    print("-" * 80)
    
    at_risk = [(team, prob) for team, prob in results['relegated_prob'].items() if prob > 0.05]
    at_risk.sort(key=lambda x: x[1], reverse=True)
    
    if at_risk:
        for team, prob in at_risk:
            exp_pts = results['expected_points'][team]
            std_pts = results['points_std'][team]
            print(f"  {team:20} {prob*100:6.2f}%  (Exp: {exp_pts:6.1f} pts, Std: {std_pts:.1f})")
    else:
        print("  No team has > 5% relegation risk")
    
    print("\n EXPECTED BOTTOM 3 FINISH")
    print("-" * 80)
    print(f"{'Team':<20} {'Exp Points':>12} {'Std Dev':>10} {'Relegation %':>12}")
    print("-" * 80)
    
    sorted_by_points = sorted(results['expected_points'].items(), key=lambda x: x[1])
    for team, exp_pts in sorted_by_points[:3]:
        std_pts = results['points_std'][team]
        prob = results['relegated_prob'][team]
        print(f"{team:<20} {exp_pts:>11.1f}  {std_pts:>9.1f}  {prob*100:>11.2f}%")

def print_top_6_finish(results, n_simulations=10000):
    print("\n" + "=" * 80)
    print("TOP 6 FINISH PROBABILITIES")
    print("=" * 80)
    
    print(f"{'Team':<20} {'Top 6 %':>10} {'Exp Points':>12} {'Std Dev':>10}")
    print("-" * 80)
    
    top6_probs = {}
    for team in results['teams']:
        t_idx = results['team_idx'][team]
        positions = results['team_positions'][:, t_idx]
        top6_count = np.sum(positions <= 6)
        top6_probs[team] = top6_count / n_simulations

    sorted_teams = sorted(top6_probs.items(), key=lambda x: x[1], reverse=True)
    for team, prob in sorted_teams:
        if prob > 0:
            exp_pts = results['expected_points'][team]
            std_pts = results['points_std'][team]
            print(f"{team:<20} {prob*100:>9.2f}%  {exp_pts:>11.1f}  {std_pts:>9.1f}")

def print_season_summary(results, n_simulations=10000):
    print("\n" + "=" * 80)
    print("SEASON SUMMARY")
    print("=" * 80)
    
    print(f"\n SIMULATIONS RUN: {n_simulations:,}")
    print("-" * 80)
    
    champion = max(results['champion_prob'].items(), key=lambda x: x[1])
    print(f"\n MOST LIKELY CHAMPION: {champion[0]} ({champion[1]*100:.2f}%)")
    
    top4 = max(results['top4_prob'].items(), key=lambda x: x[1])
    print(f" MOST LIKELY TOP 4: {top4[0]} ({top4[1]*100:.2f}%)")
    
    relegated = max(results['relegated_prob'].items(), key=lambda x: x[1])
    print(f"  MOST LIKELY RELEGATED: {relegated[0]} ({relegated[1]*100:.2f}%)")
    
    print("\n EXPECTED POINTS TABLE (Top 10)")
    print("-" * 80)
    sorted_points = sorted(results['expected_points'].items(), key=lambda x: x[1], reverse=True)
    for i, (team, exp_pts) in enumerate(sorted_points[:10]):
        std_pts = results['points_std'][team]
        champ_prob = results['champion_prob'][team]
        top4_prob = results['top4_prob'][team]
        releg_prob = results['relegated_prob'][team]
        
        print(f"{i+1:2}. {team:<20} {exp_pts:>8.1f} pts  ({std_pts:>5.1f} std)")
        print(f"       Champ: {champ_prob*100:5.2f}%  Top4: {top4_prob*100:5.2f}%  Rel: {releg_prob*100:5.2f}%")
    
    print("\n Saving detailed relegation analysis...")
    relegation_counts = {}
    for team in results['relegated_prob'].keys():
        prob = results['relegated_prob'][team]
        relegation_counts[team] = int(prob * n_simulations)
    
    relegation_df = pd.DataFrame({
        'Team': list(relegation_counts.keys()),
        'RelegatedCount': list(relegation_counts.values()),
        'RelegationProb': [results['relegated_prob'][team] for team in results['relegated_prob']],
        'ExpectedPoints': [results['expected_points'][team] for team in results['expected_points']],
        'PointsStd': [results['points_std'][team] for team in results['points_std']]
    })
    
    output_path = Path("results") / "relegation_details.csv"
    relegation_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    top4_counts = {}
    for team in results['top4_prob'].keys():
        top4_counts[team] = int(results['top4_prob'][team] * n_simulations)
    
    top4_df = pd.DataFrame({
        'Team': list(top4_counts.keys()),
        'Top4Count': list(top4_counts.values()),
        'Top4Prob': [results['top4_prob'][team] for team in results['top4_prob']],
        'ExpectedPoints': [results['expected_points'][team] for team in results['expected_points']],
        'PointsStd': [results['points_std'][team] for team in results['points_std']]
    })
    
    output_path = Path("results") / "top4_details.csv"
    top4_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

def analyze_specific_team(team_name, results, n_simulations):
    if team_name not in results['team_idx']:
        print(f"Team {team_name} not found.")
        return
        
    t_idx = results['team_idx'][team_name]
    positions = results['team_positions'][:, t_idx]
    
    relegation_sims = np.where(positions >= 18)[0]
    relegation_count = len(relegation_sims)
    
    avg_pos = np.mean(positions)
    best_pos = np.min(positions)
    worst_pos = np.max(positions)
    
    print("\n" + "=" * 80)
    print(f"TEAM SPECIFIC SIMULATION ANALYSIS: {team_name}")
    print("=" * 80)
    print(f"Total Simulations Relegated: {relegation_count} out of {n_simulations} ({relegation_count/n_simulations*100:.2f}%)")
    print(f"Average League Position: {avg_pos:.1f}")
    print(f"Best Finish: {best_pos}")
    print(f"Worst Finish: {worst_pos}")
    
    if relegation_count > 0:
        print(f"\nSimulation IDs where {team_name} got relegated (showing up to first 25):")
        print(relegation_sims[:25].tolist())
        
    position_counts = pd.Series(positions).value_counts().sort_index()
    print(f"\nPositional Distribution for {team_name}:")
    for pos, count in position_counts.items():
        print(f"  Position {pos:2d}: {count:5d} times ({count/n_simulations*100:.2f}%)")

if __name__ == "__main__":
    print("=" * 80)
    print("PREMIER LEAGUE SEASON SIMULATOR")
    print("=" * 80)
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"
    results_dir = Path("results")
    
    results_dir.mkdir(exist_ok=True)
    
    hist_path = processed_dir / "matches_features.csv"
    season_path = raw_dir / "season_2526.csv"
    
    print(f"\nReading historical data from: {hist_path}")
    hist_df = pd.read_csv(hist_path)
    hist_df['is_fixture'] = False
    
    print(f"Reading current season data from: {season_path}")
    season_df = pd.read_csv(season_path)
    
    matches_df = pd.concat([hist_df, season_df], ignore_index=True)
    
    available_seasons = sorted(matches_df['Season'].unique().tolist())
    print(f"\nAvailable seasons: {available_seasons}")
    
    if '2425' in available_seasons or '2024-25' in available_seasons:
        train_seasons = [s for s in available_seasons if s not in ['2425', '2024-25', '2526', '2025-26']]
        test_season = '2425' if '2425' in available_seasons else '2024-25'
    else:
        train_seasons = available_seasons[:-1] if len(available_seasons) > 1 else available_seasons
        test_season = available_seasons[-1] if len(available_seasons) > 0 else None

    print(f"Training seasons: {train_seasons}")
    print(f"Test season: {test_season}")
    
    if test_season:
        train_df = matches_df[~matches_df["Season"].isin([test_season])]
        test_df = matches_df[matches_df["Season"] == test_season].dropna(subset=["FTHG", "FTAG"])
    else:
        train_df = matches_df.dropna(subset=["FTHG", "FTAG"])
        test_df = pd.DataFrame()
    
    print(f"\nTraining on {len(train_df)} matches")
    print(f"Testing on {len(test_df)} matches")
    
    print("\n" + "-" * 80)
    print("FITTING INITIAL MODEL ON HISTORICAL DATA")
    print("-" * 80)
    
    model = DixonColesModel()
    model.fit(train_df)
    
    recent_seasons = ['2324', '2425', '2526']
    recent_seasons_filtered = [s for s in available_seasons if s in recent_seasons]
    
    if recent_seasons_filtered:
        recent_df = matches_df[matches_df["Season"].isin(recent_seasons_filtered)].dropna(subset=["FTHG", "FTAG"])
        print(f"\nRefitting on {len(recent_df)} recent matches (seasons: {recent_seasons_filtered})")
        model.fit(recent_df)
    
    played_df = season_df[season_df['is_fixture'] == False].dropna(subset=["FTHG", "FTAG"])
    fixtures_df = season_df[season_df['is_fixture'] == True].dropna(subset=["HomeTeam", "AwayTeam"])
    
    print(f"\nPlayed matches (2526): {len(played_df)}")
    print(f"Remaining fixtures (2526): {len(fixtures_df)}")
    
    print("\n" + "-" * 80)
    print("BUILDING CURRENT LEAGUE TABLE")
    print("-" * 80)
    
    current_table = build_current_season_table(played_df)
    print(current_table.to_string(index=False))
    
    print("\n" + "-" * 80)
    n_sims = 10000
    print(f"RUNNING {n_sims:,} SIMULATIONS...")
    print("-" * 80)
    
    results = simulate_season(model, fixtures_df, current_table, n_simulations=n_sims)
    
    print_simulation_results(results, n_sims)
    print_relegation_details(results, n_simulations=n_sims)
    print_top_6_finish(results, n_simulations=n_sims)
    print_season_summary(results, n_simulations=n_sims)
    
    analyze_specific_team("Tottenham", results, n_sims)
    
    results_df = pd.DataFrame({
        'Team': list(results['champion_prob'].keys()),
        'ChampionProb': [results['champion_prob'][team] for team in results['champion_prob']],
        'Top4Prob': [results['top4_prob'][team] for team in results['top4_prob']],
        'RelegationProb': [results['relegated_prob'][team] for team in results['relegated_prob']],
        'ExpectedPoints': [results['expected_points'][team] for team in results['expected_points']],
        'PointsStd': [results['points_std'][team] for team in results['points_std']]
    })
    
    results_df = results_df.sort_values('ChampionProb', ascending=False)
    
    output_path = results_dir / "simulation_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)