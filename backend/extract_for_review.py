import pandas as pd
import numpy as np
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Phonetic Latin junk
    text = re.sub(r'एचटीटीपी[^\s]*|डब्ल्यूडब्ल्यूडब्ल्यू[^\s]*|डॉट\s*कॉम', ' ', text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df = pd.read_csv("data/data.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["text"] = df["text"].apply(clean_text)
df = df[~df["text"].str.strip().eq("")].copy()

# Find conflicting labels
conflicting = df.groupby("text")["label"].nunique().reset_index(name="num_labels")
conflicting_texts = conflicting[conflicting["num_labels"] > 1]["text"].tolist()

conflicts_df = df[df["text"].isin(conflicting_texts)].sort_values(by="text")
conflicts_df.to_csv("conflicting_duplicates.csv", index=False)

# Short texts
df["calculated_word_count"] = df["text"].str.split().str.len()
short_texts = df[df["calculated_word_count"] <= 5].copy()

short_texts.to_csv("short_texts.csv", index=False)

print(f"Found {len(conflicting_texts)} conflicting texts.")
print(f"Found {len(short_texts)} short texts (<= 5 words).")

# Also generate 60 samples for label verification
sample_0 = df[df["label"] == 0].sample(30, random_state=42)
sample_1 = df[df["label"] == 1].sample(30, random_state=42)
pd.concat([sample_0, sample_1]).to_csv("label_verification_sample.csv", index=False)
print("Generated label_verification_sample.csv")
