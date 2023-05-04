from datasets import load_dataset

dataset = load_dataset("lca0503/soxdata_small_encodec")

print(dataset['train'][0])
