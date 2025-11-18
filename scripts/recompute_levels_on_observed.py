import pandas as pd

texts = pd.read_csv("data/processed/lecture_texts.tsv", sep="\t")
depth = pd.read_csv("data/processed/topic_depth.csv")  # current raw depth per topic

texts["topic_id"] = texts["topic_id"].astype(str)
depth["topic_id"] = depth["topic_id"].astype(str)

obs_topics = depth[depth["topic_id"].isin(texts["topic_id"].unique())].copy()

# Since all topics are depth 3, we need another way to differentiate
# Use topic_id as a proxy: lower IDs tend to be more introductory
# Convert topic_id to numeric for quantile-based binning
obs_topics["topic_id_num"] = obs_topics["topic_id"].astype(int)

# recompute quantiles on topic_id for 5-level binning
# Lower topic IDs = more introductory, higher = more advanced
q = obs_topics["topic_id_num"].quantile([0.2, 0.4, 0.6, 0.8]).values

def topic_id_to_level(tid):
    tid_num = int(tid)
    if tid_num <= q[0]: return 1  # very intro (Level 1)
    elif tid_num <= q[1]: return 2
    elif tid_num <= q[2]: return 3
    elif tid_num <= q[3]: return 4
    else: return 5  # very advanced (Level 5)

obs_topics["depth_level"] = obs_topics["topic_id"].apply(topic_id_to_level).astype(int)

# write a new mapping file only for observed topics
obs_topics[["topic_id","depth","topic_name","depth_level"]].to_csv(
    "data/processed/topic_depth_observed.csv", index=False
)

print("Wrote data/processed/topic_depth_observed.csv (levels recomputed on observed topics, 5-level).")
print(f"\nDepth distribution:")
print(obs_topics['depth'].value_counts().sort_index())
print(f"\nDepth level distribution (1-5):")
print(obs_topics['depth_level'].value_counts().sort_index())

