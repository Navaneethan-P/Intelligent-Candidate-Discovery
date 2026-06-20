import pandas as pd
import sys
import os

def validate_submission(csv_path):
    print(f"Validating submission file: {csv_path}")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File {csv_path} not found.")
        return False
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error: Could not read CSV file. {e}")
        return False
    
    all_passed = True
    
    # Check row count
    if len(df) != 100:
        print(f"❌ Row count: Expected exactly 100 rows, found {len(df)}")
        all_passed = False
    else:
        print(f"✅ Row count: {len(df)} candidates (correct)")
        
    # Check required columns
    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Columns: Missing required columns: {missing_cols}")
        all_passed = False
    else:
        print(f"✅ Columns: All required columns present ({', '.join(expected_cols)})")
        
    # Check for missing values
    if df.isnull().values.any():
        null_counts = df.isnull().sum()
        print(f"❌ Missing values: {null_counts[null_counts > 0].to_dict()}")
        all_passed = False
    else:
        print(f"✅ Missing values: None detected")
        
    # Check rank ordering
    if not df['rank'].is_monotonic_increasing:
        print(f"❌ Rank ordering: Not strictly monotonically increasing")
        all_passed = False
    else:
        print(f"✅ Rank ordering: Correctly ordered (1 to {len(df)})")
    
    # Check candidate ID format
    if 'candidate_id' in df.columns:
        invalid_ids = df[~df['candidate_id'].str.match(r'^CAND_\d{7}$')]
        if len(invalid_ids) > 0:
            print(f"❌ Candidate IDs: {len(invalid_ids)} invalid format(s)")
            all_passed = False
        else:
            print(f"✅ Candidate IDs: All match CAND_XXXXXXX format")
    
    # Check for duplicate candidate IDs
    if 'candidate_id' in df.columns:
        dupes = df['candidate_id'].duplicated().sum()
        if dupes > 0:
            print(f"❌ Duplicates: {dupes} duplicate candidate IDs")
            all_passed = False
        else:
            print(f"✅ Duplicates: None")
    
    # Check score range
    if 'score' in df.columns:
        min_score = df['score'].min()
        max_score = df['score'].max()
        print(f"📊 Score range: {min_score:.4f} to {max_score:.4f}")
    
    # Check reasoning length
    if 'reasoning' in df.columns:
        avg_len = df['reasoning'].str.len().mean()
        min_len = df['reasoning'].str.len().min()
        print(f"📊 Reasoning: avg {avg_len:.0f} chars, min {min_len} chars")
    
    print("=" * 50)
    if all_passed:
        print("🎉 SUCCESS: Submission file passes all validation checks!")
    else:
        print("⚠️  FAILED: Fix the issues above before submitting.")
    
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "outputs", "team_submission.csv")
        
    validate_submission(path)
