from datasets import load_dataset


def download_dataset(num_complaints: int = 50):
    dataset = load_dataset("milesbutler/consumer_complaints", split="train")
    pd_dataset = dataset.to_pandas()
    return pd_dataset.head(num_complaints)
