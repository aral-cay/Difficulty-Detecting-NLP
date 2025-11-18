import argparse
import pandas as pd
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="data/processed/lecture_texts.tsv")
    ap.add_argument("--depth", default="data/processed/topic_depth_observed.csv")
    ap.add_argument("--output", default="data/processed/lecture_depth_dataset.csv")
    args = ap.parse_args()

    # Read the text data
    print(f"Reading text data from {args.texts}...")
    texts_df = pd.read_csv(args.texts, sep='\t')
    print(f"  Loaded {len(texts_df)} text records")
    
    # Read the depth data
    print(f"Reading depth data from {args.depth}...")
    depth_df = pd.read_csv(args.depth)
    print(f"  Loaded {len(depth_df)} topic records")
    
    # Convert topic_id to string for consistent joining
    texts_df['topic_id'] = texts_df['topic_id'].astype(str)
    depth_df['topic_id'] = depth_df['topic_id'].astype(str)
    
    # Join on topic_id
    print("Joining texts and depth data...")
    result_df = texts_df.merge(depth_df, on='topic_id', how='inner')
    print(f"  Joined to {len(result_df)} records")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Select and reorder columns
    df = result_df[["text", "topic_id", "topic_name", "depth", "depth_level", "filepath"]]
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nWrote data/processed/lecture_depth_dataset.csv with depth_level ∈ {{1..5}}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns.tolist())}")

if __name__ == "__main__":
    main()

