import pandas as pd
import numpy as np

updates = """
02/05/2026 12:30,Aston Villa,Tottenham,1 - 2
04/05/2026 15:00,Chelsea,Nottingham Forest,1 - 3
04/05/2026 20:00,Everton,Manchester City,3 - 3
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
    if not line.strip(): continue
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
print("Updated season_2526.csv with the final matches of Gameweek 36")
