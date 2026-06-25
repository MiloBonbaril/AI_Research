from datasets import load_dataset

# 103M tokens, ~500MB, parfait pour le POC
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", num_proc=8)

# Splits déjà propres : train / validation / test
train_data = ds["train"]
val_data   = ds["validation"]  # pour ton eval loop
test_data  = ds["test"]        # touche-y seulement à la fin