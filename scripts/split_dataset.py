import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/lecture_depth_dataset_chunks.csv")
    ap.add_argument("--output_prefix", default="data/processed/lecture_depth")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    print(f"Reading chunks from {args.src}...")
    df = pd.read_csv(args.src)
    print(f"  Loaded {len(df)} chunks")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Split indices
    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    
    # Save splits
    train_path = f"{args.output_prefix}_train.csv"
    val_path = f"{args.output_prefix}_val.csv"
    test_path = f"{args.output_prefix}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSplit dataset:")
    print(f"  Train: {len(train_df)} chunks ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} chunks ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} chunks ({len(test_df)/n*100:.1f}%)")
    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

if __name__ == "__main__":
    main()

