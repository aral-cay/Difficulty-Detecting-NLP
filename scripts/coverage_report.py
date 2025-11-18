import pandas as pd

texts = pd.read_csv("data/processed/lecture_texts.tsv", sep="\t")
depth = pd.read_csv("data/processed/topic_depth.csv")  # has topic_id, depth, topic_name, depth_level (1..5)

texts["topic_id"] = texts["topic_id"].astype(str)
depth["topic_id"]  = depth["topic_id"].astype(str)

# limit depth to only topics that we actually have text for
have = depth[depth["topic_id"].isin(texts["topic_id"].unique())].copy()

# counts per topic
files_per_topic = texts.groupby("topic_id").size().rename("file_count").reset_index()
have = have.merge(files_per_topic, on="topic_id", how="left").fillna({"file_count":0})

print("\n== Raw depths among topics we actually have ==")
print(sorted(have["depth"].unique()))

print("\n== Level distribution among existing labeled rows (if already joined) ==")

try:
    df = pd.read_csv("data/processed/lecture_depth_dataset.csv")
    print(df["depth_level"].value_counts().sort_index())
except:
    print("No joined dataset yet (that's okay).")

print("\n== Top topics by file_count (head) ==")
print(have.sort_values("file_count", ascending=False)[["topic_id","topic_name","depth","file_count"]].head(20).to_string(index=False))

print("\n== Depth → topic examples ==")
for d in sorted(have["depth"].unique()):
    ex = have[have["depth"]==d].head(5)[["topic_id","topic_name","file_count"]]
    print(f"\nDepth {d} examples:\n", ex.to_string(index=False))

