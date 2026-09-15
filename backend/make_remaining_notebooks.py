import json
import os

def create_notebook(filename, cells_config):
    cells = []
    for cell_type, source in cells_config:
        cells.append({
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + '\n' for line in source.strip().split('\n')]
        })
    
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

# ==========================================
# Notebook 3: Baselines
# ==========================================
nb3_cells = [
    ("markdown", "# Step 4: Baseline Model (TF-IDF + Logistic Regression)\nThis baseline gives us a fast, cheap model to set the bar for the transformer."),
    ("code", """import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns"""),
    ("code", """# Load splits
try:
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    test_df = pd.read_csv("data/processed/test.csv")
except FileNotFoundError:
    print("Processed data not found. Please run 02_Cleaning_Split.ipynb first.")
    raise
"""),
    ("code", """# TF-IDF Vectorization
# Using a small manual Hindi stopword list since there is no standard NLTK one readily available
hindi_stopwords = ["के", "में", "की", "है", "और", "को", "से", "हैं", "कि", "का", "लिए", "पर", "यह"] 

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=hindi_stopwords, max_features=10000)
X_train = vectorizer.fit_transform(train_df['text'].fillna(""))
X_val = vectorizer.transform(val_df['text'].fillna(""))

y_train = train_df['label']
y_val = val_df['label']"""),
    ("code", """# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_val)

acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Macro-F1: {f1:.4f}")
print("\\nClassification Report:")
print(classification_report(y_val, y_pred))"""),
    ("code", """# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()""")
]

# ==========================================
# Notebook 4: Tokenization Analysis
# ==========================================
nb4_cells = [
    ("markdown", "# Step 5: Tokenizer-aware length analysis\nHere we pick MuRIL and tokenize the full text column to find the ideal `max_length`."),
    ("code", """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer"""),
    ("code", """# Load full data
df = pd.read_csv("data/data.csv").dropna(subset=['text'])

# Load Tokenizer (MuRIL)
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)"""),
    ("code", """# Tokenize and get lengths
def get_token_length(text):
    # .encode gives the token IDs (including special tokens [CLS] and [SEP])
    return len(tokenizer.encode(str(text), add_special_tokens=True))

print("Tokenizing all texts... This may take a minute.")
df['token_length'] = df['text'].apply(get_token_length)"""),
    ("code", """# Stats
desc = df['token_length'].describe(percentiles=[0.9, 0.95, 0.99])
print(desc)

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(df['token_length'], bins=50, kde=True)
plt.title(f'Token Length Distribution ({model_name})')
plt.xlabel('Number of Tokens')
plt.show()"""),
    ("code", """# Truncation Analysis
truncated_256 = (df['token_length'] > 256).sum()
truncated_512 = (df['token_length'] > 512).sum()
total = len(df)

print(f"Samples truncated at 256 tokens: {truncated_256} ({(truncated_256/total)*100:.2f}%)")
print(f"Samples truncated at 512 tokens: {truncated_512} ({(truncated_512/total)*100:.2f}%)")""")
]

# ==========================================
# Notebook 5: Transformer Fine Tuning
# ==========================================
nb5_cells = [
    ("markdown", "# Step 6: Fine-tune the Transformer Baseline\nUsing HuggingFace Trainer to fine-tune MuRIL on the cleaned splits, with class-weighted loss and evaluating on Macro-F1."),
    ("code", """import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support"""),
    ("code", """# Load datasets
train_df = pd.read_csv("data/processed/train.csv").dropna(subset=['text'])
val_df = pd.read_csv("data/processed/validation.csv").dropna(subset=['text'])
test_df = pd.read_csv("data/processed/test.csv").dropna(subset=['text'])"""),
    ("code", """# Model prep
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Based on Step 5, set MAX_LEN appropriately (e.g. 256)
MAX_LEN = 256 

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)"""),
    ("code", """# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}"""),
    ("code", """# Handle Class Imbalance with Custom Trainer
# Calculate class weights (inverse of frequency)
class_weights = (1 - (train_df['label'].value_counts().sort_index() / len(train_df))).values
class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss"""),
    ("code", """# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",      # evaluate each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir='./logs',
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)"""),
    ("code", """# Train
print("Starting Training...")
trainer.train()"""),
    ("code", """# Save Best Model
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
print("Best model saved to ./best_model")""")
]

# ==========================================
# Notebook 6: Error Analysis
# ==========================================
nb6_cells = [
    ("markdown", "# Step 7: Error Analysis\nPull the 30 validation examples the model was most confident about but got wrong."),
    ("code", """import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification"""),
    ("code", """# Load validation data and best model
val_df = pd.read_csv("data/processed/validation.csv").dropna(subset=['text'])
model_path = "./best_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print("Could not load model. Ensure Step 6 completed and saved the model to ./best_model")
    raise

model.eval()
if torch.cuda.is_available():
    model.to('cuda')"""),
    ("code", """# Predict and get probabilities
texts = val_df['text'].tolist()

all_probs = []
all_preds = []

batch_size = 32
print("Running inference on validation set...")
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

val_df['predicted_label'] = all_preds
val_df['prob_0'] = [p[0] for p in all_probs]
val_df['prob_1'] = [p[1] for p in all_probs]"""),
    ("code", """# Find confident errors
errors = val_df[val_df['label'] != val_df['predicted_label']].copy()
errors['confidence'] = errors[['prob_0', 'prob_1']].max(axis=1)

# Top 30 most confident errors
top_errors = errors.sort_values(by='confidence', ascending=False).head(30)
print(f"Found {len(errors)} total errors.")
print("\\nTop 30 Most Confident Errors:")
display(top_errors[['text', 'label', 'predicted_label', 'confidence']])"""),
    ("markdown", "### Categorization Task\nPlease manually categorize these 30 examples into:\n1. **Label Noise** (the true label is actually wrong)\n2. **Ambiguous Content** (hard to tell even for a human)\n3. **Register/Style Overfitting** (e.g. looks like a real news article but is fake)\n4. **Genuinely hard cases**")
]

create_notebook("03_Baselines.ipynb", nb3_cells)
create_notebook("04_Tokenization_Analysis.ipynb", nb4_cells)
create_notebook("05_Transformer_FineTuning.ipynb", nb5_cells)
create_notebook("06_Error_Analysis.ipynb", nb6_cells)

print("Created notebooks 03_Baselines.ipynb, 04_Tokenization_Analysis.ipynb, 05_Transformer_FineTuning.ipynb, 06_Error_Analysis.ipynb")
