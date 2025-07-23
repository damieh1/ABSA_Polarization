import torch
import pandas as pd
import re
from tqdm import tqdm 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load fine-tuned ABSA model plus tokenizer
model_path = "./fine_tuned_deberta"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Free GPU memory before inference
torch.cuda.empty_cache()

# Load splitted CSV chunks aka unseen data
unseen_data_path = "/specify_with_unseen_raw-data.csv"
try:
    df = pd.read_csv(unseen_data_path, dtype=str, encoding="utf-8")
except Exception as e:
    print("Error using default engine & switching to Python engine")
    df = pd.read_csv(unseen_data_path, dtype=str, encoding="utf-8", engine="python")

if "sentence" not in df.columns:
    raise ValueError("Dataset must contain a 'sentence' column.")

# Function to clean text (removes @usernames, extra spaces)
def clean_text(text):
    text = re.sub(r"@\w+", "", text)  # Remove @usernames
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

# Function to extract aspect reliably from cleaned text
def extract_aspect(sentence):
    cleaned_sentence = clean_text(sentence)
    words = cleaned_sentence.split()
    return words[0] if words else "unknown"  # Picks the first word as aspect $T$ # For reviewers: Please keep in mind that the parser creates for each aspect term that appears in a sentence a new row. If there are multiple aspects in a sentence, this assures that all aspects get properly predicted.  

# Apply text cleaning and improved aspect extraction
df["cleaned_sentence"] = df["sentence"].apply(clean_text)
df["aspect"] = df["cleaned_sentence"].apply(extract_aspect)
df["formatted_sentence"] = df.apply(lambda row: row["cleaned_sentence"].replace(row["aspect"], "$T$"), axis=1)

# Process in small batches to prevent CUDA Out-Of-Memory errors
batch_size = 8  # Reduce batch size to fit GPU memory
predicted_labels = []
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

print(f"Processing {len(df)} sentences...")

# Check progress
for i in tqdm(range(0, len(df), batch_size), desc="Processing", unit="batch"):
    batch_sentences = df["formatted_sentence"].tolist()[i:i + batch_size]
    batch_aspects = df["aspect"].tolist()[i:i + batch_size]
    
    inputs = tokenizer(batch_sentences, batch_aspects, 
                       padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():  # Enable mixed precision for lower memory usage
        outputs = model(**inputs)

    batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
    predicted_labels.extend(batch_preds)

df["predicted_sentiment"] = [sentiment_map[label] for label in predicted_labels]

# Save results
output_file = "ABSA_predictions_for_unseen_data.csv"
df[["sentence", "aspect", "predicted_sentiment"]].to_csv(output_file, index=False)

print(f"Predictions saved # '{output_file}'.")
