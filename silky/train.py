import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np

class training_data():
    option = [
              "Maysee/tiny-imagenet",
              "",
              "path/to/data"
             ]
    choice = 0
    hf_repo = "Maysee/tiny-imagenet"
    repo_url = ""
    repo_path = "path/to/data"

def train(trainer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = trainer.option[trainer.choice]
    dataset = datasets.load_dataset(repo, split="train")
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_siz=4)
    for batch in dataloader:
        print(batch)
    return


