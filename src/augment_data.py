import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.ImageProcessor import ImageProcessor

df = pd.read_csv(r"data/train.csv", index_col=0)

# group by classId and get the number of images per class
groups = dict()
for group, group_df in df.groupby("classId"):
    groups[group] = len(group_df)

# compute mean of images over all classes
mean = int(np.mean(list(groups.values())).round(0))

# Display class distribution before augmentation
fig, axes = plt.subplots(figsize=(10, 10))
axes.bar(groups.keys(), groups.values())
axes.hlines(
    y=np.mean(list(groups.values())),
    xmin=0,
    xmax=43,
    color="red",
    linestyles="dashed",
    label="Mean",
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
axes.set_title("Class distribution before augmentation")
axes.set_xlabel("Class")
axes.set_ylabel("Number of images")
axes.legend()
fig.savefig(os.path.join("images", "class_distribution_before_augmentation.png"))


# Perform augmentation
processor = ImageProcessor()
augmented_df = pd.DataFrame()

for group, group_df in tqdm(df.groupby("classId")):
    # processing underrepresented classes
    if len(group_df) < mean:
        # mulitply images by 3 using rotation, zoom and noise
        if len(group_df) <= 500:
            for path in group_df["Path"]:
                test = processor.augment_image(
                    path, rotation=True, zoom=True, noise=True
                )
                augmented_df = pd.concat([augmented_df, test]).reset_index(drop=True)
        # mulitply images by 2 using rotation and zoom
        elif len(group_df) > 500:
            for path in group_df["Path"]:
                test = processor.augment_image(
                    path, rotation=True, zoom=True, noise=False
                )
                augmented_df = pd.concat([augmented_df, test]).reset_index(drop=True)

    # processing overrepresented classes (delete random images)
    else:
        if np.abs(len(group_df) - mean) >= 700:
            images = list()
            for i in range(400):
                path = np.random.choice(group_df["Path"])
                new_path = path.replace("Train", "Bkp")
                if not os.path.exists("/".join(new_path.split("/")[:-1])):
                    os.makedirs("/".join(new_path.split("/")[:-1]))
                if path.replace("Train", "Bkp") not in os.listdir(
                    "/".join(new_path.split("/")[:-1])
                ):
                    shutil.copy(path, new_path)
                    row = df[df["Path"] == path].index
                    df = df.drop(index=row.values)
                else:
                    continue

new_df = pd.concat([df, augmented_df])


# Display class distribution after augmentation
for group, group_df in new_df.groupby("classId"):
    groups[group] = len(group_df)

# create plot
fig, axes = plt.subplots(figsize=(10, 10))
axes.bar(groups.keys(), groups.values())
axes.hlines(y=mean, xmin=0, xmax=43, color="red", linestyles="dashed", label="Mean")
axes.set_title("Class distribution after augmentation")
axes.set_xlabel("Class")
axes.set_ylabel("Number of images")
axes.legend()

fig.savefig(os.path.join("images", "class_distribution_after_augmentation.png"))


# Save changes to csv
new_df = new_df.sort_values(by=["classId", "Path"]).reset_index(drop=True)

new_df.to_csv(r"data/augmented_train.csv")
