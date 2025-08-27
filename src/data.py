from datasets import load_dataset

def load_cronicas_dataset(path: str):
    raw = load_dataset("json", data_files=path, split="train")
    return raw