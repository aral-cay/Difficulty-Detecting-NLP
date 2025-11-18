import argparse
import pandas as pd
import os

def collapse_level(level):
    """Collapse 5-level to 3-level: 1→Beginner, 2-3→Intermediate, 4-5→Advanced
    
    This gives more balanced distribution:
    - Beginner: Level 1 (818 files)
    - Intermediate: Levels 2-3 (273 files) 
    - Advanced: Levels 4-5 (739 files)
    """
    if level == 1:
        return 1  # Beginner
    elif level in [2, 3]:
        return 2  # Intermediate
    else:  # level >= 4
        return 3  # Advanced

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/lecture_depth_train.csv")
    ap.add_argument("--val", default="data/processed/lecture_depth_val.csv")
    ap.add_argument("--test", default="data/processed/lecture_depth_test.csv")
    args = ap.parse_args()

    print("Collapsing 5-level labels to 3-level (1→Beginner, 2-3→Intermediate, 4-5→Advanced)...")
    
    for split in ["train", "val", "test"]:
        input_file = getattr(args, split)
        if not os.path.exists(input_file):
            print(f"  Skipping {split}: {input_file} not found")
            continue
        
        df = pd.read_csv(input_file)
        print(f"\n{split.upper()}:")
        print(f"  Original levels: {df['depth_level'].value_counts().sort_index().to_dict()}")
        
        # Collapse levels
        df['depth_level'] = df['depth_level'].apply(collapse_level).astype(int)
        
        print(f"  New levels: {df['depth_level'].value_counts().sort_index().to_dict()}")
        
        # Save
        output_file = input_file.replace("lecture_depth_", "lecture_depth3_")
        df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
    
    print("\n✅ Relabeling complete!")

if __name__ == "__main__":
    main()

