import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer,
    TrainingArguments, get_scheduler, AutoConfig
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Load Hugging Face token
# For Reviewers: You need an Huggingface Account to run the 
with open("hf_token.txt", "r") as token_file:
    hf_token = token_file.read().strip()

# Load Ground_truth_APC.raw
def load_apc_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sentences, aspects, sentiments = [], [], []
    for i in range(0, len(lines), 3):
        sentences.append(lines[i].strip())
        aspects.append(lines[i + 1].strip())
        sentiments.append(lines[i + 2].strip())
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    labels = [sentiment_map[s] for s in sentiments]
    df = pd.DataFrame({"sentence": sentences, "aspect": aspects, "labels": labels})
    df["aspect"] = df["aspect"].str.lower().str.strip()
    return df

dataset_path = "/projects/replicate_the_paper/training/data_input/Ground_truth_APC.raw"
df_apc = load_apc_dataset(dataset_path)

# Shuffle and split
df_apc = df_apc.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * len(df_apc))
df_apc_train = df_apc.iloc[:train_size]
df_apc_valid = df_apc.iloc[train_size:]

# Mixup augmentation
def mixup_data(df, alpha=0.2):
    indices = np.random.permutation(len(df))
    lambda_ = np.random.beta(alpha, alpha)
    df["sentence"] = df["sentence"].astype(str) + " " + df.iloc[indices]["sentence"].astype(str)
    df["labels"] = (lambda_ * df["labels"] + (1 - lambda_) * df.iloc[indices]["labels"]).astype(int)
    return df

df_apc_train = mixup_data(df_apc_train)

# Tokenizer and model
model_name = "yangheng/deberta-v3-large-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
if "$T$" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["$T$"])
    print("Added `$T$` to tokenizer vocabulary.")

config = AutoConfig.from_pretrained(model_name, 
                                    hidden_dropout_prob=0.3, 
                                    attention_probs_dropout_prob=0.3, 
                                    num_labels=3)

model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, token=hf_token)
model.resize_token_embeddings(len(tokenizer))  # Always resize in case $T$ was added

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["sentence"], examples["aspect"], padding="max_length", truncation=True)

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_apc_train).map(tokenize_function, batched=True),
    "validation": Dataset.from_pandas(df_apc_valid).map(tokenize_function, batched=True)
})

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.cuda.empty_cache()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
)

# Scheduler
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
    num_warmup_steps=0,
    num_training_steps=len(dataset["train"])
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-5), lr_scheduler),
    compute_metrics=compute_metrics,  # <<== ? Metrics now included
)

torch.cuda.empty_cache()
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_deberta")
tokenizer.save_pretrained("./fine_tuned_deberta")

print("? Fine-tuning complete. Model and tokenizer saved.")

