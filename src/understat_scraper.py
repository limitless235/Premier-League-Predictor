import pandas as pd
from pathlib import Path
import time
from understatapi import UnderstatClient

def download_understat_data():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    seasons = range(2014, 2025)
    
    with UnderstatClient() as understat:
        for year in seasons:
            print(f"Fetching {year}...")
            
            try:
                match_data = understat.league(league="EPL").get_match_data(season=str(year))
                
                matches = []
                for item in match_data:
                    try:
                        home_goals = int(item["goals"]["h"]) if item["goals"]["h"] is not None else None
                        away_goals = int(item["goals"]["a"]) if item["goals"]["a"] is not None else None
                        home_xg = float(item["xG"]["h"]) if item["xG"]["h"] is not None else None
                        away_xg = float(item["xG"]["a"]) if item["xG"]["a"] is not None else None
                        
                        row = {
                            "datetime": item.get("datetime"),
                            "home": item["h"]["title"],
                            "away": item["a"]["title"],
                            "homegoals": home_goals,
                            "awaygoals": away_goals,
                            "home_xG": home_xg,
                            "away_xG": away_xg,
                            "data_quality_flag": True
                        }
                        matches.append(row)
                    except (TypeError, KeyError, ValueError):
                        continue
                
                df = pd.DataFrame(matches)
                if len(df) == 0:
                    print(f"✗ No matches found for {year}")
                    continue
                    
                output_file = data_dir / f"understat_{year}.csv"
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {year} ({len(df)} matches) to {output_file}")
                
            except Exception as e:
                print(f"✗ Error processing {year}: {e}")
                
            time.sleep(2)

if __name__ == "__main__":
    download_understat_data()