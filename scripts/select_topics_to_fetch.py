import pandas as pd

alldata = pd.read_csv("data/raw/LectureBank/alldata.tsv", sep="\t")
depth   = pd.read_csv("data/processed/topic_depth.csv")

# Find the topic column
col_map = {c.lower(): c for c in alldata.columns}
topic_col = col_map.get("topic") or col_map.get("topic_id") or col_map.get("topicid")
if not topic_col:
    topic_col = alldata.columns[0]  # fallback

alldata["Topic"] = alldata[topic_col].astype(str)
depth["topic_id"] = depth["topic_id"].astype(str)

# which topics have URLs in the metadata?
topics_with_urls = set(alldata["Topic"].unique())
cand = depth[depth["topic_id"].isin(topics_with_urls)].copy()

# weight by depth extremes: take N from min depth and N from max depth
N = 30
dmin, dmax = cand["depth"].min(), cand["depth"].max()
tmin = cand[cand["depth"]==dmin].head(N)["topic_id"].tolist()
tmax = cand[cand["depth"]==dmax].head(N)["topic_id"].tolist()

print("Shallow topics to fetch:", " ".join(tmin))
print("Deep topics to fetch:", " ".join(tmax))

# Also show some stats
print(f"\nDepth range in available topics: {dmin} to {dmax}")
print(f"Shallow topics ({len(tmin)}): {tmin[:5]}...")
print(f"Deep topics ({len(tmax)}): {tmax[:5]}...")

