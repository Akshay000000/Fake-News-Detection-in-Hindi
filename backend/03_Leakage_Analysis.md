# Step 3: Leakage & Duplicate Analysis
Before splitting the data, we must ensure there are no exact or near-duplicates that cross splits, and identify any label conflicts.



```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
# Load Data
df = pd.read_csv("data/data.csv")
df = df.dropna(subset=['text']).copy()
print(f"Original shape: {df.shape}")

```

    Original shape: (17123, 4)
    

### 1. Exact Duplicates & Conflicting Labels



```python
# Exact Duplicates
exact_dupes = df[df.duplicated(subset=['text'], keep=False)]
print(f"Number of exact duplicate texts: {len(exact_dupes)}")

# Label Conflicts (same text, different label)
grouped = df.groupby('text')['label'].nunique()
conflicting_texts = grouped[grouped > 1].index
print(f"Number of texts with conflicting labels: {len(conflicting_texts)}")

if len(conflicting_texts) > 0:
    display(df[df['text'].isin(conflicting_texts)].sort_values(by='text').head(10))

```

    Number of exact duplicate texts: 297
    Number of texts with conflicting labels: 1
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>label</th>
      <th>wcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8757</th>
      <td>8757</td>
      <td>पालघर की तर्ज पर पंजाब के होशियार पुर में संत ...</td>
      <td>1</td>
      <td>43</td>
    </tr>
    <tr>
      <th>9414</th>
      <td>9414</td>
      <td>पालघर की तर्ज पर पंजाब के होशियार पुर में संत ...</td>
      <td>0</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>


### 2. Near-Duplicates (TF-IDF Char n-grams + Cosine Similarity)



```python
# Using Char n-grams to find near-duplicates (e.g. spelling changes, minor words added/removed)
# We will use a sample if the dataset is too large, but 17k rows is manageable for a sparse matrix
print("Vectorizing for near-duplicate detection...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=10000)
X = vectorizer.fit_transform(df['text'])

```

    Vectorizing for near-duplicate detection...
    


```python
# Calculate similarities (this can take a moment for 17k x 17k)
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

```

    Finding near-duplicate pairs (Similarity > 0.90)...
    

    Found 1527 near-duplicate pairs.
    


```python
# Analyze Near-Duplicates
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
    
    print("\nSample of near-duplicates (Similarity > 0.90):")
    for _, row in nd_df.head(3).iterrows():
        print("SIMILARITY:", row['similarity'])
        print("TEXT 1:", df.iloc[int(row['idx1'])]['text'][:200], "...")
        print("TEXT 2:", df.iloc[int(row['idx2'])]['text'][:200], "...")
        print("-" * 50)

```

    Near-duplicate pairs with conflicting labels: 39
    
    Sample of near-duplicates (Similarity > 0.90):
    SIMILARITY: 1.000000000000002
    TEXT 1: रिपोर्ट मित्र इस पूरे समय मध्य नाम से काम कर रहा है कैलाबास सीए - यह जानकर आश्चर्य हुआ कि छह वर्षों में वे एक-दूसरे को जानते थे किसी भी समय यह बात सामने नहीं आई 25 वर्षीय स्थानीय महिला लुसी रीड ने मंग ...
    TEXT 2: रिपोर्ट मित्र इस पूरे समय मध्य नाम से काम कर रहा है कैलाबास सीए - यह जानकर आश्चर्य हुआ कि छह वर्षों में वे एक-दूसरे को जानते थे किसी भी समय यह बात सामने नहीं आई 25 वर्षीय स्थानीय महिला लुसी रीड ने मंग ...
    --------------------------------------------------
    SIMILARITY: 1.0000000000000018
    TEXT 1: ईमेल  “हमें अधिक से अधिक नौकरियों की आवश्यकता है। यहां बेरोज़गारी बहुत ज़्यादा है” उन्होंने कहा। उसने दुनिया में जितने भी लोगों को मैं जानता हूं उससे कहीं अधिक कर्मचारियों अधिक लोगों को काम पर रखा है। ...
    TEXT 2: ईमेल  “हमें अधिक से अधिक नौकरियों की आवश्यकता है। यहां बेरोज़गारी बहुत ज़्यादा है” उन्होंने कहा। उसने दुनिया में जितने भी लोगों को मैं जानता हूं उससे कहीं अधिक कर्मचारियों अधिक लोगों को काम पर रखा है। ...
    --------------------------------------------------
    SIMILARITY: 1.0000000000000016
    TEXT 1: रिपोर्ट मित्र इस पूरे समय मध्य नाम से काम कर रहा है कैलाबास सीए - यह जानकर आश्चर्य हुआ कि छह वर्षों में वे एक-दूसरे को जानते थे किसी भी समय यह बात सामने नहीं आई 25 वर्षीय स्थानीय महिला लुसी रीड ने मंग ...
    TEXT 2: रिपोर्ट मित्र इस पूरे समय मध्य नाम से काम कर रहा है कैलाबास सीए - यह जानकर आश्चर्य हुआ कि छह वर्षों में वे एक-दूसरे को जानते थे किसी भी समय यह बात सामने नहीं आई 25 वर्षीय स्थानीय महिला लुसी रीड ने मंग ...
    --------------------------------------------------
    

### 3. Cleaning the Data based on Leakage Analysis
We should drop exact duplicates (resolving conflicts by dropping or keeping majority vote), and consider dropping one from each near-duplicate pair before moving to splitting.



```python
# Drop exact duplicates (for conflicts, we can just drop both or keep first - let's drop duplicates entirely for safety)
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

```

    Shape after dropping exact duplicates: (16826, 4)
    Shape after dropping near-duplicates: (16470, 4)
    

    Saved cleaned data to data_cleaned_noleakage.csv
    
