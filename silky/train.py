import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from . import model
import os

class training_options():
    option = [
              "Maysee/tiny-imagenet",
              "",
              "path/to/data"
             ]
    choice = 0
    hf_repo = "Maysee/tiny-imagenet"
    repo_url = ""
    repo_path = "path/to/data"

def train(repo):
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if repo == "" : repo = "Maysee/tiny-imagenet"
    dataset = datasets.load_dataset(repo, split="train")
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=4)

    logfilename = repo + "-run"
    logging.basicConfig(level=logging.DEBUG, filename=logfilename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    savepath = "./saves/winners"
    progpath = "./saves/in-prog"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    mdl = model.velvet()
    mdl.create(64, 64, 64, 500, 200, 0)
    wins = 0
    for batch in dataloader:
        for item in batch:
            tally = np.zeros(200)
            for i in range(0, 200):
                output = np.array(mdl.update(item["image"]))
                tally = tally + output
            guess = np.unravel_index(np.argmax(tally))
            answer = item["label"]
            logging.info('{ "guess" : "' + guess + '" }')
            logging.info('{ "answer"  : "' + answer + '" }')
            if answer == guess:
                wins += 1
                logging.info('WIN!')
                mdl.save(savepath)
            else:
                mdl.permute(20)
    mdl.save(progpath)
    return
    


