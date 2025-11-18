import argparse, os, re, time, pathlib, mimetypes
import pandas as pd
import requests
from urllib.parse import urlparse
from tqdm import tqdm

def safe(name: str) -> str:
    return re.sub(r'[^-\w.,() ]+', '_', name)[:180]

def guess_ext(resp, url):
    ctype = resp.headers.get('Content-Type', '')
    ext = mimetypes.guess_extension((ctype.split(';')[0] if ctype else '').strip()) or ''
    if not ext:
        ext = pathlib.Path(urlparse(url).path).suffix
    return ext or '.html'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alldata", default="data/raw/LectureBank/alldata.tsv")
    ap.add_argument("--outdir",   default="data/downloads")
    ap.add_argument("--topics",   nargs="*", help="Topic IDs to keep (e.g. 2 21 233)")
    ap.add_argument("--max",      type=int, default=0, help="Max files (0=all)")
    ap.add_argument("--timeout",  type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.alldata, sep="\t")
    col = {c.lower(): c for c in df.columns}
    id_col    = col.get("id") or list(df.columns)[0]
    url_col   = col.get("url")
    topic_col = col.get("topic") or col.get("topic_id") or col.get("topicid")
    title_col = col.get("title") or col.get("name")

    if url_col is None:
        raise ValueError("Couldn't find a URL column in alldata.tsv")

    keep = df
    if args.topics and topic_col:
        keep = keep[keep[topic_col].astype(str).isin(args.topics)]

    os.makedirs(args.outdir, exist_ok=True)
    n = 0
    skipped = 0
    for _, row in tqdm(keep.iterrows(), total=len(keep)):
        url = str(row[url_col])
        if not url:
            continue
        topic_id = str(row[topic_col]) if topic_col else "unknown"
        title = safe(str(row[title_col])) if title_col else f"lecture_{row[id_col]}"
        dest_dir = os.path.join(args.outdir, topic_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Check if file already exists (skip existing)
        # Try common extensions
        file_exists = False
        for ext in ['.pdf', '.pptx', '.html', '.htm']:
            fname = f"{row[id_col]}_{title}{ext}"
            fpath = os.path.join(dest_dir, fname)
            if os.path.exists(fpath):
                skipped += 1
                file_exists = True
                break
        
        if file_exists:
            continue
        
        # File doesn't exist, download it
        try:
            with requests.get(url, timeout=args.timeout, allow_redirects=True, stream=True, headers={"User-Agent":"lecture-depth/1.0"}) as r:
                if r.status_code != 200:
                    continue
                ext = guess_ext(r, url)
                fname = f"{row[id_col]}_{title}{ext}"
                fpath = os.path.join(dest_dir, fname)
                with open(fpath, "wb") as f:
                    for chunk in r.iter_content(1024*64):
                        if chunk: f.write(chunk)
            n += 1
            if args.max and n >= args.max:
                break
            time.sleep(0.25)
        except Exception:
            continue
    
    print(f"\nDownloaded {n} new files, skipped {skipped} existing files")

if __name__ == "__main__":
    main()
