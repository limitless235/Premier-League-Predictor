import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from scipy.stats import poisson
import pathlib
import math
import time

def dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    if home_goals == 0 and away_goals == 0:
        return 1 - (lambda_home * lambda_away * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (lambda_home * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (lambda_away * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def log_likelihood(params, home_idx, away_idx, fthg, ftag, n_teams):
    attack = params[:n_teams]
    defence = params[n_teams:2*n_teams]
    home_advantage = params[2*n_teams]
    rho = params[2*n_teams + 1]
    
    lambda_home = np.exp(attack[home_idx] + defence[away_idx] + home_advantage)
    lambda_away = np.exp(attack[away_idx] + defence[home_idx])
    
    tau = np.ones_like(fthg, dtype=float)
    
    mask_00 = (fthg == 0) & (ftag == 0)
    mask_01 = (fthg == 0) & (ftag == 1)
    mask_10 = (fthg == 1) & (ftag == 0)
    mask_11 = (fthg == 1) & (ftag == 1)
    
    tau[mask_00] = 1 - (lambda_home[mask_00] * lambda_away[mask_00] * rho)
    tau[mask_01] = 1 + (lambda_home[mask_01] * rho)
    tau[mask_10] = 1 + (lambda_away[mask_10] * rho)
    tau[mask_11] = 1 - rho
    
    tau = np.clip(tau, 1e-10, np.inf)
    
    poisson_home = poisson.pmf(fthg, lambda_home)
    poisson_away = poisson.pmf(ftag, lambda_away)
    
    poisson_home = np.clip(poisson_home, 1e-10, np.inf)
    poisson_away = np.clip(poisson_away, 1e-10, np.inf)

    log_likelihood_sum = np.sum(np.log(tau) + np.log(poisson_home) + np.log(poisson_away))

    return -log_likelihood_sum

class DixonColesModel:
    def __init__(self):
        self.teams = None
        self.attack = {}
        self.defence = {}
        self.home_advantage = None
        self.rho = None
        self.fitted = False
    
    def fit(self, matches_df):
        df = matches_df.dropna(subset=["FTHG", "FTAG"]).copy()
        self.teams = sorted(df["HomeTeam"].unique().tolist() + df["AwayTeam"].unique().tolist())
        self.teams = sorted(list(set(self.teams)))
        n_teams = len(self.teams)
        
        league_avg_scored = df["FTHG"].mean()
        league_avg_conceded = df["FTAG"].mean()
        
        attack_init = []
        defence_init = []
        for team in self.teams:
            home_scored = df[df["HomeTeam"]==team]["FTHG"].mean()
            away_scored = df[df["AwayTeam"]==team]["FTAG"].mean()
            team_avg_scored = np.nanmean([home_scored, away_scored])
            attack_init.append(np.log(team_avg_scored / league_avg_scored + 0.01))
            
            home_conceded = df[df["HomeTeam"]==team]["FTAG"].mean()
            away_conceded = df[df["AwayTeam"]==team]["FTHG"].mean()
            team_avg_conceded = np.nanmean([home_conceded, away_conceded])
            defence_init.append(-np.log(team_avg_conceded / league_avg_conceded + 0.01))
        
        params_init = (attack_init + defence_init + [0.25, -0.1])
        params_init = np.array(params_init)
        
        bounds = [(0, 0)] + [(-3, 3)] * (n_teams - 1)
        bounds += [(-3, 3)] * n_teams
        bounds += [(0, 1), (-0.99, 0.99)]
        
        team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        home_idx_arr = df['HomeTeam'].map(team_to_idx).values
        away_idx_arr = df['AwayTeam'].map(team_to_idx).values
        fthg_arr = df['FTHG'].values
        ftag_arr = df['FTAG'].values

        result = minimize(
            log_likelihood,
            params_init,
            args=(home_idx_arr, away_idx_arr, fthg_arr, ftag_arr, n_teams),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000}
        )
        
        self.attack = dict(zip(self.teams, result.x[:n_teams]))
        self.defence = dict(zip(self.teams, result.x[n_teams:2*n_teams]))
        self.home_advantage = result.x[-2]
        self.rho = result.x[-1]
        self.fitted = True
        
        print(f"Optimization success: {result.success}")
        print(f"Learned rho: {self.rho:.4f}")
        print(f"Learned home_advantage: {self.home_advantage:.4f}")
        
        attack_sorted = sorted(self.attack.items(), key=lambda x: x[1], reverse=True)
        defence_sorted = sorted(self.defence.items(), key=lambda x: x[1])
        
        print("\nAll Attack Ratings (descending):")
        for team, val in attack_sorted:
            print(f"  {team}: {val:.4f}")
        
        print("\nAll Defence Ratings (ascending = better defence):")
        for team, val in defence_sorted:
            print(f"  {team}: {val:.4f}")
    
    def predict_match(self, home_team, away_team, max_goals=6):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if home_team not in self.teams or away_team not in self.teams:
            return {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33, 'lambda_home': 1.0, 'lambda_away': 1.0}
        
        lambda_home = np.exp(self.attack[home_team] + self.defence[away_team] + self.home_advantage)
        lambda_away = np.exp(self.attack[away_team] + self.defence[home_team])
        
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                tau = dixon_coles_tau(h, a, lambda_home, lambda_away, self.rho)
                poisson_h = np.exp(-lambda_home) * (lambda_home ** h) / math.factorial(h)
                poisson_a = np.exp(-lambda_away) * (lambda_away ** a) / math.factorial(a)
                probs[h, a] = tau * poisson_h * poisson_a
        
        probs = probs / np.sum(probs)
        
        home_win = np.sum(np.tril(probs, -1))
        draw = np.sum(np.diag(probs))
        away_win = np.sum(np.triu(probs, 1))
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away
        }
    
    def predict_batch(self, fixtures_df):
        print("\nStarting predictions...")
        predictions = []
        start_time = time.time()
        
        for idx, row in fixtures_df.iterrows():
            pred = self.predict_match(row['HomeTeam'], row['AwayTeam'])
            pred['HomeTeam'] = row['HomeTeam']
            pred['AwayTeam'] = row['AwayTeam']
            predictions.append(pred)
            
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Completed {idx + 1}/{len(fixtures_df)} matches ({elapsed:.1f}s)")

        df_result = pd.DataFrame(predictions)
        print(f"Predictions completed in {time.time() - start_time:.1f}s")
        return df_result

if __name__ == "__main__":
    hist_path = pathlib.Path("data/processed/matches_features.csv")
    curr_path = pathlib.Path("data/raw/season_2526.csv")
    
    try:
        hist_df = pd.read_csv(hist_path)
        hist_df['is_fixture'] = False
        curr_df = pd.read_csv(curr_path)
        matches_df = pd.concat([hist_df, curr_df], ignore_index=True)
    except FileNotFoundError:
        print("Data files not found.")
        exit(1)
    
    print(f"\nAvailable Season values in matches_df: {sorted(matches_df['Season'].unique().tolist())}")
    print(f"Season column dtype: {matches_df['Season'].dtype}")
    
    df = matches_df
    available_seasons = sorted(matches_df['Season'].unique().tolist())
    print(f"Available seasons: {available_seasons}")
    
    if 2425 in available_seasons or '2425' in available_seasons or '2024-25' in available_seasons:
        train_seasons = [s for s in available_seasons if s not in [2425, '2425', '2024-25', 2526, '2526', '2025-26']]
        test_season = 2425 if 2425 in available_seasons else ('2425' if '2425' in available_seasons else '2024-25')
    else:
        train_seasons = available_seasons[:-1] if len(available_seasons) > 1 else available_seasons
        test_season = available_seasons[-1] if len(available_seasons) > 0 else None
    
    print(f"Training seasons: {train_seasons}")
    print(f"Test season: {test_season}")
    
    if test_season:
        train_df = df[~df["Season"].isin([test_season])]
        test_df = df[df["Season"] == test_season].dropna(subset=["FTHG", "FTAG"])
    else:
        train_df = df.dropna(subset=["FTHG", "FTAG"])
        test_df = pd.DataFrame()
    
    print(f"\nTraining on {len(train_df)} matches")
    print(f"Testing on {len(test_df)} matches")
    
    if len(train_df) == 0:
        print("ERROR: No training data available!")
        exit(1)
    
    model = DixonColesModel()
    model.fit(train_df)
    
    if len(test_df) > 0:
        test_predictions = model.predict_batch(test_df)
        
        test_df = test_df.reset_index(drop=True)
        
        test_df['pred_H'] = test_predictions['home_win']
        test_df['pred_D'] = test_predictions['draw']
        test_df['pred_A'] = test_predictions['away_win']
        
        test_df['predicted_result'] = test_df[['pred_H', 'pred_D', 'pred_A']].idxmax(axis=1).str[-1]
        
        test_df['actual_result'] = test_df.apply(
            lambda row: 'H' if row['FTHG'] > row['FTAG'] else ('D' if row['FTHG'] == row['FTAG'] else 'A'),
            axis=1
        )
        
        accuracy = (test_df['predicted_result'] == test_df['actual_result']).mean()
        print(f"Test set accuracy: {accuracy:.4f}")
        
        y_true = test_df['actual_result'].map({'H': 0, 'D': 1, 'A': 2}).values
        y_pred = test_df[['pred_H', 'pred_D', 'pred_A']].values
        
        try:
            log_loss_value = log_loss(y_true, y_pred, labels=[0, 1, 2])
            print(f"Test set log loss: {log_loss_value:.4f}")
        except Exception as e:
            print(f"Could not calculate log loss: {e}")
    else:
        print("\nNo test matches found. Model trained on historical data only.")
        print("Consider adjusting Season filtering in data files.")