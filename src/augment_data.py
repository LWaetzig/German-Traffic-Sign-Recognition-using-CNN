import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.ImageProcessor import ImageProcessor


df = pd.read_csv(r"data/train.csv", index_col=0)

# Display class distribution before augmentation
groups = dict()

for group, group_df in df.groupby("classId"):
    groups[group] = len(group_df)
mean = int(np.mean(list(groups.values())).round(0))

fig, axes = plt.subplots(figsize=(10, 10))
axes.bar(groups.keys(), groups.values())
axes.hlines(
    y=np.mean(list(groups.values())), xmin=0, xmax=43, color="red", linestyles="dashed"
)
for i, value in enumerate(groups.values()):
    if value < np.mean(list(groups.values())):
        axes.vlines(
            x=i,
            ymin=value,
            ymax=np.mean(list(groups.values())),
            color="black",
            linestyles="dashed",
        )


# Perform augmentation
processor = ImageProcessor()
augmented_df = pd.DataFrame()

for group, group_df in tqdm(df.groupby("classId")):
    if len(group_df) < mean:
        if len(group_df) <= 500:
            for path in group_df["Path"]:
                test = processor.augment_image(
                    path, rotation=True, zoom=True, noise=True
                )
                augmented_df = pd.concat([augmented_df, test]).reset_index(drop=True)
        elif len(group_df) > 500:
            for path in group_df["Path"]:
                test = processor.augment_image(
                    path, rotation=True, zoom=True, noise=False
                )
                augmented_df = pd.concat([augmented_df, test]).reset_index(drop=True)
    else:
        continue

new_df = pd.concat([df, augmented_df])


# Display class distribution after augmentation
for group, group_df in new_df.groupby("classId"):
    groups[group] = len(group_df)

fig, axes = plt.subplots(figsize=(10, 10))
axes.bar(groups.keys(), groups.values())
axes.hlines(
    y=np.mean(list(groups.values())), xmin=0, xmax=43, color="red", linestyles="dashed"
)
for i, value in enumerate(groups.values()):
    if value < np.mean(list(groups.values())):
        axes.vlines(
            x=i,
            ymin=value,
            ymax=np.mean(list(groups.values())),
            color="black",
            linestyles="dashed",
        )
