import pandas as pd
import numpy as np
import requests
from pathlib import Path
import io
import argparse

name_map = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham",
    "Leeds United": "Leeds",
    "Sunderland AFC": "Sunderland"
}

def fetch_2526_data(max_gw=None):
    # url = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv"
    # headers = {"User-Agent": "Mozilla/5.0"}
    # response = requests.get(url, headers=headers)
    
    with open('/Users/limitless/.gemini/antigravity/brain/68859ca1-e5c5-4b7f-8063-438230b331e6/.system_generated/steps/57/content.md', 'r') as f:
        content = f.read()
    csv_text = content.split("---", 1)[1].strip()
    df = pd.read_csv(io.StringIO(csv_text))
    
    df['HomeTeam'] = df['Home Team'].replace(name_map)
    df['AwayTeam'] = df['Away Team'].replace(name_map)
    df['Season'] = '2526'
    
    # --- MANUAL SCORE OVERRIDES ---
    # fixturedownload.com is missing this completed GW32 match
    override_idx = (df['HomeTeam'] == 'Manchester United') & (df['AwayTeam'] == 'Leeds') & (df['Round Number'].astype(str) == '32')
    df.loc[override_idx, 'Result'] = '1 - 2'
    # ------------------------------
    
    df['is_fixture'] = df['Result'].isna() | (df['Result'].astype(str).str.strip() == '')
    
    if max_gw is not None:
        df.loc[df['Round Number'] > max_gw, 'is_fixture'] = True
        df.loc[df['Round Number'] > max_gw, 'Result'] = np.nan
    
    scores = df['Result'].str.extract(r'(\d+)\s*-\s*(\d+)')
    df['FTHG'] = pd.to_numeric(scores[0])
    df['FTAG'] = pd.to_numeric(scores[1])
    
    conditions = [
        df['FTHG'] > df['FTAG'],
        df['FTHG'] < df['FTAG'],
        df['FTHG'] == df['FTAG']
    ]
    choices = ['H', 'A', 'D']
    df['FTR'] = np.select(conditions, choices, default=None)
    
    df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season', 'is_fixture']]
    
    output_path = Path('data/raw/season_2526.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    played = len(df[df['is_fixture'] == False])
    remaining = len(df[df['is_fixture'] == True])
    
    teams_list = df['HomeTeam'].dropna().unique().tolist() + df['AwayTeam'].dropna().unique().tolist()
    teams = sorted(list(set(teams_list)))
    
    print(f"Total rows: {len(df)}")
    print(f"Played matches: {played}")
    print(f"Remaining fixtures: {remaining}")
    print(f"Unique teams: {teams}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch current season data")
    parser.add_argument("--max-gw", type=int, default=None, help="Max gameweek to include actual results for (e.g., 32). Games after this will be treated as fixtures to simulate.")
    args = parser.parse_args()
    
    fetch_2526_data(max_gw=args.max_gw)