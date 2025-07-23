import os
import pandas as pd

# Split JSON-files from folder into chunks of 80k 
input_folder = "/replicate_the_paper/biaffine/raw_data/"
output_folder = "/replicate_the_paper/biaffine/raw_data/splitted/"
chunk_size = 80000

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".jsonl"):
        continue

    file_path = os.path.join(input_folder, filename)
    print(f"\n? Processing: {filename}")

    # Read JSONL
    try:
        chunk_iter = pd.read_json(file_path, lines=True, chunksize=chunk_size)
    except Exception as e:
        print(f"Failed {filename}: {e}")
        continue

    # Split and save chunks
    for i, chunk in enumerate(chunk_iter):
        base_name = os.path.splitext(filename)[0]
        chunk_label = f"{base_name}_part{i+1}"
        chunk["chunk_label"] = chunk_label

        output_path = os.path.join(output_folder, f"{chunk_label}.jsonl")
        chunk.to_json(output_path, orient="records", lines=True)
        print(f"Chunk: # {output_path} ({len(chunk)} rows)")

