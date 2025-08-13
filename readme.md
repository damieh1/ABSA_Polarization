# ABSA Sentiment Analysis: Scripts for Aspect-Based Classification and Evaluation

- Repository contains scripts and resources for fine-tuning a DeBERTa-based model for aspect-based sentiment analysis (ABSA)
- The project includes preprocessing, fine-tuning, evaluation, and application to unseen data.
- It was developed as part of a study analyzing sentiment in user-generated content about (geo-)political entities, also known as aspect terms, in social media discourse.

---

## Folder Structure

```
├── format_apc.py                  # Formats the aspect-polarity classification input
├── eval_per_aspect.py             # Evaluates the trained model per aspect
├── fine_tune_deberta.py           # Fine-tunes DeBERTa on ABSA data
├── prepare_unseen_absa_data.py    # Applies trained model to new data
├── replicate_the_paper/
│   ├── biaffine_dep_parser.py     # Runs dependency parsing to extract aspects
│   ├── spitter_jsonl.py           # Converts structured input into JSONL
│   └── training/data_input/
│       ├── Ground_truth_APC.csv   # Annotated dataset for training
│       └── Ground_truth_APC.raw   # Raw dataset for formatting
└── requirements.txt               # Python dependencies
```

---

## Getting Started

### 1. Environment Setup
Make sure you’re using **Python 3.10+**. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Format Training Data
Build triplets before training: Use `format_apc.py` to convert `.raw` and `.csv` files into the proper format for Hugging Face model training.
```bash
python format_apc.py --input_csv training/data_input/Ground_truth_APC.csv --output_jsonl formatted_data.jsonl
```

### 3. Fine-Tune the Model
Train the DeBERTa model using `fine_tune_deberta.py`. This will save the model into the `checkpoints` folder.
```bash
python fine_tune_deberta.py --train_file formatted_data.jsonl --output_dir checkpoints/absa_model
```

### 4. Evaluate the Model
Run `eval_per_aspect.py` to compute precision, recall, and F1-score per aspect.
```bash
python eval_per_aspect.py --eval_file formatted_data.jsonl --model_dir checkpoints/absa_model
```

### 5. Inference on Unseen Data
Use `prepare_unseen_absa_data.py` to classify sentiments in new datasets.
```bash
python prepare_unseen_absa_data.py --input new_data.csv --model_dir checkpoints/absa_model
```

### 6. Dependency Parsing
Aspect spans and their dependencies were extracted using `biaffine_dep_parser.py`, based on a Biaffine dependency parser (trained on English Treebanks). Output is aligned to the sentiment classification task and formatted using `spitter_jsonl.py`.

---

## Requirements (See requirements.txt)
```
transformers==4.39.3
datasets==2.18.0
torch>=2.0
pandas>=1.5.3
scikit-learn>=1.1.3
tqdm>=4.64.1
nltk>=3.8.1
emoji>=2.8.0
```

---

## License
This repository is distributed under the MIT License.

---
