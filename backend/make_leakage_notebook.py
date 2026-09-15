import json
import os

def create_notebook(filename, cells_config):
    cells = []
    for cell_type, source in cells_config:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + '\n' for line in source.strip().split('\n')]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

nb_cells = [
    ("markdown", "# Step 3: Leakage & Duplicate Analysis\nBefore splitting the data, we must ensure there are no exact or near-duplicates that cross splits, and identify any label conflicts."),
    ("code", """import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
"""),
    ("code", """# Load Data
df = pd.read_csv("data/data.csv")
df = df.dropna(subset=['text']).copy()
print(f"Original shape: {df.shape}")
"""),
    ("markdown", "### 1. Exact Duplicates & Conflicting Labels"),
    ("code", """# Exact Duplicates
exact_dupes = df[df.duplicated(subset=['text'], keep=False)]
print(f"Number of exact duplicate texts: {len(exact_dupes)}")

# Label Conflicts (same text, different label)
grouped = df.groupby('text')['label'].nunique()
conflicting_texts = grouped[grouped > 1].index
print(f"Number of texts with conflicting labels: {len(conflicting_texts)}")

if len(conflicting_texts) > 0:
    display(df[df['text'].isin(conflicting_texts)].sort_values(by='text').head(10))
"""),
    ("markdown", "### 2. Near-Duplicates (TF-IDF Char n-grams + Cosine Similarity)"),
    ("code", """# Using Char n-grams to find near-duplicates (e.g. spelling changes, minor words added/removed)
# We will use a sample if the dataset is too large, but 17k rows is manageable for a sparse matrix
print("Vectorizing for near-duplicate detection...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=10000)
X = vectorizer.fit_transform(df['text'])
"""),
    ("code", """# Calculate similarities (this can take a moment for 17k x 17k)
# We will do it in chunks to avoid memory explosion if necessary, or just compute the full matrix
from sklearn.metrics.pairwise import linear_kernel
import gc

# To avoid massive memory usage (O(N^2)), let's do a fast chunked search
threshold = 0.90
near_dupe_pairs = []

print("Finding near-duplicate pairs (Similarity > 0.90)...")
chunk_size = 2000
for i in range(0, X.shape[0], chunk_size):
    end_i = min(i + chunk_size, X.shape[0])
    chunk = X[i:end_i]
    # Compute similarity of chunk against ALL data
    sims = linear_kernel(chunk, X)
    
    # Find indices where sim > threshold
    for i_idx, row in enumerate(sims):
        global_i = i + i_idx
        # Only look at pairs (global_i, j) where global_i < j to avoid symmetric duplicates
        for j in np.where(row > threshold)[0]:
            if global_i < j:
                near_dupe_pairs.append((global_i, j, row[j]))
    
    del sims
    gc.collect()

print(f"Found {len(near_dupe_pairs)} near-duplicate pairs.")
"""),
    ("code", """# Analyze Near-Duplicates
if len(near_dupe_pairs) > 0:
    nd_df = pd.DataFrame(near_dupe_pairs, columns=['idx1', 'idx2', 'similarity'])
    nd_df = nd_df.sort_values(by='similarity', ascending=False)
    
    # Are there near-duplicates with conflicting labels?
    labels1 = df.iloc[nd_df['idx1']]['label'].values
    labels2 = df.iloc[nd_df['idx2']]['label'].values
    nd_df['label1'] = labels1
    nd_df['label2'] = labels2
    nd_df['conflict'] = nd_df['label1'] != nd_df['label2']
    
    print(f"Near-duplicate pairs with conflicting labels: {nd_df['conflict'].sum()}")
    
    print("\\nSample of near-duplicates (Similarity > 0.90):")
    for _, row in nd_df.head(3).iterrows():
        print("SIMILARITY:", row['similarity'])
        print("TEXT 1:", df.iloc[int(row['idx1'])]['text'][:200], "...")
        print("TEXT 2:", df.iloc[int(row['idx2'])]['text'][:200], "...")
        print("-" * 50)
"""),
    ("markdown", "### 3. Cleaning the Data based on Leakage Analysis\nWe should drop exact duplicates (resolving conflicts by dropping or keeping majority vote), and consider dropping one from each near-duplicate pair before moving to splitting."),
    ("code", """# Drop exact duplicates (for conflicts, we can just drop both or keep first - let's drop duplicates entirely for safety)
clean_df = df.drop_duplicates(subset=['text'], keep=False)
print(f"Shape after dropping exact duplicates: {clean_df.shape}")

# Optional: drop near-duplicates (keeping one of the pair)
if len(near_dupe_pairs) > 0:
    to_drop = set()
    for _, row in nd_df.iterrows():
        # Keep idx1, drop idx2
        if row['idx1'] not in to_drop:
            to_drop.add(row['idx2'])
    
    # Translate original positional indices to dataframe index
    indices_to_drop = clean_df.index.intersection(df.iloc[list(to_drop)].index)
    clean_df = clean_df.drop(indices_to_drop)

print(f"Shape after dropping near-duplicates: {clean_df.shape}")

# Save the final cleaned data before splitting
clean_df.to_csv("data/data_cleaned_noleakage.csv", index=False)
print("Saved cleaned data to data_cleaned_noleakage.csv")
""")
]

create_notebook("03_Leakage_Analysis.ipynb", nb_cells)
print("Created 03_Leakage_Analysis.ipynb")
