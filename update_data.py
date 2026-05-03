import pandas as pd
import numpy as np

updates = """
18/04/2026 12:30,Brentford,Fulham,0 - 0
18/04/2026 15:00,Leeds,Wolves,3 - 0
18/04/2026 15:00,Newcastle,Bournemouth,1 - 2
18/04/2026 17:30,Spurs,Brighton,2 - 2
18/04/2026 20:00,Chelsea,Man Utd,0 - 1
19/04/2026 14:00,Aston Villa,Sunderland,4 - 3
19/04/2026 14:00,Everton,Liverpool,1 - 2
19/04/2026 14:00,Nott'm Forest,Burnley,4 - 1
19/04/2026 16:30,Man City,Arsenal,2 - 1
20/04/2026 20:00,Crystal Palace,West Ham,0 - 0
21/04/2026 20:00,Brighton,Chelsea,3 - 0
22/04/2026 20:00,Bournemouth,Leeds,2 - 2
22/04/2026 20:00,Burnley,Man City,0 - 1
24/04/2026 20:00,Sunderland,Nott'm Forest,0 - 5
25/04/2026 12:30,Fulham,Aston Villa,1 - 0
25/04/2026 15:00,Liverpool,Crystal Palace,3 - 1
25/04/2026 15:00,West Ham,Everton,2 - 1
25/04/2026 15:00,Wolves,Spurs,0 - 1
25/04/2026 17:30,Arsenal,Newcastle,1 - 0
27/04/2026 20:00,Man Utd,Brentford,2 - 1
01/05/2026 20:00,Leeds,Burnley,3 - 1
02/05/2026 15:00,Brentford,West Ham,3 - 0
02/05/2026 15:00,Newcastle,Brighton,3 - 1
02/05/2026 15:00,Wolves,Sunderland,1 - 1
02/05/2026 17:30,Arsenal,Fulham,3 - 0
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
print("Updated season_2526.csv")
