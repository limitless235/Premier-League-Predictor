import pandas as pd

def check_features():
    df = pd.read_csv('data/processed/matches_features.csv')
    
    # Print ALL column names one per line
    for col in df.columns:
        print(col)
    
    # Print total number of columns
    print(f"Total columns: {len(df.columns)}")
    
    # Print number of null values per column, only showing columns that have any nulls
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"{col}: {count}")
    
    # Print total rows
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    check_features()
