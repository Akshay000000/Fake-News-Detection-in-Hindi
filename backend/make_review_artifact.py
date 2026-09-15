import pandas as pd
import os

conflicts = pd.read_csv("conflicting_duplicates.csv")
short_texts = pd.read_csv("short_texts.csv")
samples = pd.read_csv("label_verification_sample.csv")

# Categorize short texts heuristically
def categorize_short(text):
    words = str(text).split()
    if len(words) <= 1:
        return 'Junk (Single word / Empty)'
    elif not any(c.isalpha() or '\u0900' <= c <= '\u097F' for c in str(text)):
        return 'Junk (Only symbols/numbers)'
    else:
        return 'Valid (Needs review)'

short_texts['Category'] = short_texts['text'].apply(categorize_short)

with open(r"C:\Users\Jatin\.gemini\antigravity-ide\brain\8e278036-72ba-4ec3-ba1f-d4d1605a426e\review_data.md", "w", encoding="utf-8") as f:
    f.write("# Step 1(a): Conflicting Label Texts\n")
    f.write("> [!IMPORTANT]\n> Please review these 8 texts that appear with BOTH labels. We plan to DROP both instances unless you state one label is definitely correct.\n\n")
    f.write(conflicts[['text', 'label']].to_markdown(index=False))
    f.write("\n\n---\n\n")

    f.write("# Step 1(b): Short Texts (≤ 5 words)\n")
    f.write("> [!NOTE]\n> Here are the short texts categorized. We plan to DROP the ones marked 'Junk' and KEEP the ones marked 'Valid'. Please review if you disagree.\n\n")
    
    junk = short_texts[short_texts['Category'].str.contains('Junk')]
    valid = short_texts[short_texts['Category'].str.contains('Valid')]
    
    f.write("### Proposed JUNK (To Drop)\n")
    if len(junk) > 0:
        f.write(junk[['text', 'label', 'Category']].to_markdown(index=False))
    else:
        f.write("None.")
    f.write("\n\n### Proposed VALID (To Keep)\n")
    f.write(valid[['text', 'label']].to_markdown(index=False))
    f.write("\n\n---\n\n")

    f.write("# Step 2: Label Verification Sample\n")
    f.write("> [!IMPORTANT]\n> Review these 60 samples. Currently, we assume 1 = FAKE and 0 = REAL. Please confirm if this is accurate based on your domain knowledge.\n\n")
    f.write(samples[['text', 'label']].to_markdown(index=False))
    
print("Artifact review_data.md generated.")
