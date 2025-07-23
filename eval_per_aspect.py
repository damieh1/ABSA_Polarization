import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# Load Model, EVAL File, Specify Length and Batch_Size
MODEL_PATH = "./fine_tuned_deberta"
EVAL_FILE = "/projects/replicate_the_paper/training/data_input/Ground_truth_APC.raw"
MAX_LENGTH = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentiment Label 
label_map = {"negative": 0, "neutral": 1, "positive": 2}
inv_label_map = {v: k for k, v in label_map.items()}

# Model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Load APC dataset
def load_apc_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return pd.DataFrame({
        "sentence": lines[0::3],
        "aspect": lines[1::3],
        "label": lines[2::3]
    })

df = load_apc_dataset(EVAL_FILE)
df["aspect"] = df["aspect"].str.lower().str.strip()  # Normalize aspects
df["label_id"] = df["label"].map(label_map)


# Tokenization
class ABSADataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            list(df["sentence"]),
            list(df["aspect"]),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.labels = torch.tensor(df["label_id"].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = ABSADataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Save predictions 
df["predicted_id"] = all_preds
df["predicted_label"] = [inv_label_map[i] for i in all_preds]

# Overall metrics
print("Classification Report")
print(classification_report(all_labels, all_preds, target_names=label_map.keys()))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.keys())
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Per-Aspect Evaluation
print("\nPer-Aspect Evaluation")
for aspect in sorted(df["aspect"].unique()):
    subset = df[df["aspect"] == aspect]
    if len(subset) < 5:
        continue
    report = classification_report(
        subset["label_id"], subset["predicted_id"],
        output_dict=True, zero_division=0
    )
    acc = np.mean(subset["label_id"] == subset["predicted_id"])
    f1 = report["weighted avg"]["f1-score"]
    recall = report["weighted avg"]["recall"]
    precision = report["weighted avg"]["precision"]
    print(f"{aspect:<15} Acc: {acc:.2f}  Prec: {precision:.2f}  Rec: {recall:.2f}  F1: {f1:.2f}")
