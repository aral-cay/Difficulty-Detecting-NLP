import argparse
import pandas as pd
import numpy as np

def compute_depth(topic_id):
    """Compute depth based on topic ID structure.
    
    Topic IDs are hierarchical:
    - 1-digit: depth 1 (e.g., 1, 2, 3)
    - 2-digit: depth 2 (e.g., 11, 21, 31)
    - 3-digit: depth 3 (e.g., 111, 211, 311)
    """
    topic_str = str(topic_id).strip()
    # Remove any non-digit characters and compute depth
    digits_only = ''.join(filter(str.isdigit, topic_str))
    if not digits_only:
        return 1
    return len(digits_only)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="data/raw/LectureBank/taxonomy.csv")
    ap.add_argument("--output", default="data/processed/topic_depth.csv")
    args = ap.parse_args()

    # Read taxonomy
    df = pd.read_csv(args.taxonomy)
    
    # Rename columns if needed (handle variations)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'topic' in col_lower and 'id' in col_lower:
            col_map[col] = 'topic_id'
        elif 'topic' in col_lower and 'id' not in col_lower:
            col_map[col] = 'topic_name'
    
    df = df.rename(columns=col_map)
    
    # Ensure we have the right columns
    if 'topic_id' not in df.columns:
        topic_id_col = [c for c in df.columns if 'id' in c.lower()][0]
        df['topic_id'] = df[topic_id_col]
    
    if 'topic_name' not in df.columns:
        topic_name_col = [c for c in df.columns if 'topic' in c.lower() and 'id' not in c.lower()][0]
        df['topic_name'] = df[topic_name_col]
    
    # Compute depth for each topic
    df['depth'] = df['topic_id'].apply(compute_depth)
    
    # 5-level binning -> labeled 1..5 (very intro -> very advanced)
    q = df["depth"].quantile([0.2, 0.4, 0.6, 0.8]).values
    
    def depth_to_level(d):
        if d <= q[0]: return 1  # very intro
        elif d <= q[1]: return 2
        elif d <= q[2]: return 3
        elif d <= q[3]: return 4
        else: return 5          # very advanced
    
    df["depth_level"] = df["depth"].apply(depth_to_level).astype(int)
    
    # (optional) remove the old 0..2 bin if you had it
    if "depth_bin" in df.columns:
        df = df.drop(columns=["depth_bin"])
    
    # Select and reorder columns
    result = df[['topic_id', 'depth', 'topic_name', 'depth_level']].copy()
    
    # Save to CSV
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    
    print(f"Computed depth for {len(result)} topics")
    print(f"Wrote data/processed/topic_depth.csv with depth_level ∈ {{1..5}}")
    print(f"\nDepth distribution:")
    print(result['depth'].value_counts().sort_index())
    print(f"\nDepth level distribution:")
    print(result['depth_level'].value_counts().sort_index())

if __name__ == "__main__":
    main()

