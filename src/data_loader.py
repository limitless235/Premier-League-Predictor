import requests
import pandas as pd
import time
import io
from pathlib import Path

def download_fdco_data():
    base_url = "https://www.football-data.co.uk/mmz4281"
    required_columns = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A"
    ]
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    seasons = [f"{year % 100:02d}{(year % 100) + 1:02d}" for year in range(2000, 2025)]
    
    for season in seasons:
        url = f"{base_url}/{season}/E0.csv"
        print(f"Downloading {season}...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns].copy()
            
            df["Season"] = season
            
            output_file = data_dir / f"fdco_{season}.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ Saved {season} to {output_file}")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to download {season}: {e}")
            continue
        except pd.errors.EmptyDataError:
            print(f"✗ Empty data for {season}")
            continue
        except Exception as e:
            print(f"✗ Error processing {season}: {e}")
            continue
        
        time.sleep(2)

if __name__ == "__main__":
    download_fdco_data()
