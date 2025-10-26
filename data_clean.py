import os
import pandas as pd
import tqdm
import time
import numpy as np
import cv2

# Read the CSV file
df = pd.read_csv('./data/DFEW/DFEW_emo_label.csv')

# Define emotion mapping
emotion_map = {
    0: 'happy',
    1: 'happy',
    2: 'sad',
    3: 'neutral',
    4: 'angry',
    5: 'surprise',
    6: 'disgust',
    7: 'fear'
}

# Map labels to actual emotion text
df['actual'] = df['label'].map(emotion_map)
df["video_id"] = df["order"]
df["file_path"] = "./data/DFEW/part_1/" + df["video_id"].astype(str) + ".mp4"


# Save the updated CSV
df.to_csv('./data/DFEW/actual_DFEW_emo_label.csv', index=False)

print("Mapping complete!")
print(f"\nFirst 10 rows:")
print(df[['order', 'label', 'actual']].head(10))
print(f"\nLabel distribution:")
print(df['actual'].value_counts().sort_index())
