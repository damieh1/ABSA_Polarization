import pandas as pd
import re

# Ground_Truth path aka Training data
file_path = "Ground_truth_new.csv"
df = pd.read_csv(file_path)

# Function to replace aspect terms e.g., Muslims with $T$
def replace_aspect(sentence, aspect):
    aspect_pattern = r'\b' + re.escape(aspect) + r'\b'
    return re.sub(aspect_pattern, '$T$', sentence, flags=re.IGNORECASE)

# Apply transformation
df["Sentence"] = df.apply(lambda row: replace_aspect(row["Sentence"], row["Aspect"]), axis=1)

# Write APC formatted file
apc_formatted_file_path = "Ground_truth_APC.raw"
with open(apc_formatted_file_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"{row['Sentence']}\n")
        f.write(f"{row['Aspect']}\n")
        f.write(f"{row['GroundTruth']}\n")

print(f"APC file was saved as {apc_formatted_file_path}")
