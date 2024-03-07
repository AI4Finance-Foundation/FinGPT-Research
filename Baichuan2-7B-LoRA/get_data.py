from datasets import load_dataset

dataset = load_dataset('FinGPT/fingpt-sentiment-train')
dataset.save_to_disk('fingpt-sentiment-train')

