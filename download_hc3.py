import os
from datasets import load_dataset

# 1. Define the local directory
SAVE_DIR = "hc3_local"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Starting download from Hugging Face...")

# 2. Load the dataset 
# We use the 'json' loader directly to bypass the 'HC3.py' script error
dataset = load_dataset(
    "json", 
    data_files="https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl",
    split="train"
)

# 3. Save to local disk in Parquet format
# This is much faster for future loads
print(f"Saving {len(dataset)} samples to {SAVE_DIR}...")
dataset.to_parquet(os.path.join(SAVE_DIR, "hc3_train.parquet"))

print("Done! You can now load this locally without an internet connection.")

# --- HOW TO LOAD IT LATER ---
# from datasets import load_dataset
# local_ds = load_dataset("parquet", data_files="hc3_local/hc3_train.parquet")