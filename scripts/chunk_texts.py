import argparse
import pandas as pd
import re

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    if pd.isna(text) or not text:
        return []
    
    # Split by sentences first, then by words
    sentences = re.split(r'[.!?]\s+', str(text))
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_size = len(words)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap (last overlap words from previous chunk)
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + words
            current_size = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_size += sentence_size
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/lecture_depth_dataset.csv")
    ap.add_argument("--output", default="data/processed/lecture_depth_dataset_chunks.csv")
    ap.add_argument("--chunk_size", type=int, default=512, help="Target words per chunk")
    ap.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks")
    args = ap.parse_args()

    print(f"Reading dataset from {args.src}...")
    df = pd.read_csv(args.src)
    print(f"  Loaded {len(df)} records")
    
    chunked_rows = []
    
    for idx, row in df.iterrows():
        chunks = chunk_text(row['text'], chunk_size=args.chunk_size, overlap=args.overlap)
        
        if not chunks:
            # If no chunks, keep original text
            chunk_row = row.to_dict()
            chunk_row['chunk_id'] = f"{idx}_0"
            chunked_rows.append(chunk_row)
        else:
            # Create one row per chunk
            for chunk_idx, chunk_content in enumerate(chunks):
                chunk_row = row.to_dict()
                chunk_row['text'] = chunk_content
                chunk_row['chunk_id'] = f"{idx}_{chunk_idx}"
                chunked_rows.append(chunk_row)
    
    result_df = pd.DataFrame(chunked_rows)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    result_df.to_csv(args.output, index=False)
    print(f"\nChunked {len(df)} records into {len(result_df)} chunks")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()

