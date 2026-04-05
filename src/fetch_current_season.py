import pandas as pd
import numpy as np
import requests
from pathlib import Path
import io

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

def fetch_2526_data():
    url = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text))
    
    df['HomeTeam'] = df['Home Team'].replace(name_map)
    df['AwayTeam'] = df['Away Team'].replace(name_map)
    df['Season'] = '2526'
    
    df['is_fixture'] = df['Result'].isna() | (df['Result'].astype(str).str.strip() == '')
    
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
    fetch_2526_data()