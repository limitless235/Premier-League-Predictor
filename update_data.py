import pandas as pd
import numpy as np

updates = """
03/05/2026 14:00,Bournemouth,Crystal Palace,3 - 0
03/05/2026 15:30,Man Utd,Liverpool,3 - 2
"""

name_map = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham"
}

df = pd.read_csv('data/raw/season_2526.csv')

for line in updates.strip().split('\n'):
    date, home, away, res = line.split(',')
    home = name_map.get(home, home)
    away = name_map.get(away, away)
    fthg, ftag = res.split(' - ')
    
    mask = (df['HomeTeam'] == home) & (df['AwayTeam'] == away)
    if mask.any():
        df.loc[mask, 'FTHG'] = int(fthg)
        df.loc[mask, 'FTAG'] = int(ftag)
        
        fthg_int = int(fthg)
        ftag_int = int(ftag)
        
        if fthg_int > ftag_int:
            ftr = 'H'
        elif fthg_int < ftag_int:
            ftr = 'A'
        else:
            ftr = 'D'
            
        df.loc[mask, 'FTR'] = ftr
        df.loc[mask, 'is_fixture'] = False
    else:
        print(f"Warning: Match not found {home} vs {away}")

df.to_csv('data/raw/season_2526.csv', index=False)
print("Updated season_2526.csv with today's matches")
