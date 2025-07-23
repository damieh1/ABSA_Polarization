from supar import Parser
import torch
import pandas as pd
from tqdm import tqdm
import csv
import os
import json

## CUDA check
# Peer reviewers: Without GPU access parsing might take up to 12-15 hours. Consider using a VM or HPC. 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = Parser.load('biaffine-dep-en', weights_only=True)
parser.model = parser.model.to(device)  # Use GPU!

# Load XLSX-Path for Entity-Dict inclduing aspect-terms
entity_df = pd.read_excel('entity_dict_18_04_2025.xlsx')
entity_dict = {}
for entity_type in entity_df.columns:
    for entity in entity_df[entity_type].dropna():
        value = str(entity).strip().lower()
        if value and value != "nan":
            entity_dict[value] = entity_type
print(f"Load {len(entity_dict)} entities.")

# Input & Output directories / Iterates through all raw_data files in folder
# Peer reviewers: Edit this if you want to replicate the results. If you need access to RAW data please: contact editors
input_folder = "/replicate_the_paper/biaffine/raw_data/splitted/"
output_folder = "/replicate_the_paper/biaffine/raw_data/parsed/"
os.makedirs(output_folder, exist_ok=True)

BATCH_SIZE = 512  # Used 512 a max batch-size for parsing

# Process each JSONL file
for filename in os.listdir(input_folder):
    if not filename.endswith(".jsonl"):
        continue

    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(output_folder, filename.replace('.jsonl', '_entities.csv'))
    print(f"\n? Processing File-# {filename}")

    found_entities = []
    batch_comments = []
    batch_data = []

    # Make sure to read file line by line
    with open(input_file, 'r') as f:
        contents = f.readlines()

    for i, content in enumerate(tqdm(contents)):
        try:
            j = json.loads(content.strip())
            if "Comment" not in j or not isinstance(j["Comment"], str) or not j["Comment"].strip():
                continue

            batch_comments.append(j["Comment"])
            batch_data.append(j)

            # Batch processing
            if len(batch_comments) >= BATCH_SIZE or i == len(contents) - 1:
                predictions = parser.predict(batch_comments, lang='en', prob=True, verbose=False, device=device)
                for jdata, predict in zip(batch_data, predictions):
                    for index, word in enumerate(predict.words):
                        if word.lower() in entity_dict:
                            found_entities.append((
                                jdata.get("Unnamed: 0"),
                                jdata.get("Username"),
                                jdata.get("VideoID"),
                                word,
                                entity_dict[word.lower()],
                                jdata.get("Comment"),
                                predict.rels[index],
                                predict.words[predict.arcs[index] - 1] if predict.arcs[index] != 0 else 'ROOT',
                                jdata.get("Timestamp")
                            ))
                batch_comments = []
                batch_data = []

        except json.JSONDecodeError:
            print(f"Identified invalid JSON entry at line {i+1}.")

    # Specify output columns
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Comment ID', 'Username', 'VideoID', 'Entity', "Category", "Comment", "Dependency Relation", "Head Word", "Timestamp"])
        writer.writerows(found_entities)
    print(f"The file was saved under {output_file}")

