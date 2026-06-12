import pandas as pd
import sys
import os

def validate_submission(csv_path):
    print(f"Validating submission file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return False
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: Could not read CSV file. {e}")
        return False
        
    if len(df) != 100:
        print(f"Error: Expected exactly 100 rows, found {len(df)}")
        return False
    else:
        print("Success: Correct row count (100 candidates).")
        
    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return False
    else:
        print("Success: Required columns present.")
        
    if df.isnull().values.any():
        print("Error: Found missing values in the submission file.")
        return False
    else:
        print("Success: No missing values detected.")
        
    if not df['rank'].is_monotonic_increasing:
        print("Error: Rank column is not strictly monotonically increasing.")
        return False
    else:
        print("Success: Ranks are correctly ordered.")
        
    print("\nSuccess: Submission file matches official specifications.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "outputs", "team_submission.csv")
        
    validate_submission(path)
