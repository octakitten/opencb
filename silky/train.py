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
    dataformat = dataset.with_format("torch", device=gpu)
    # dataloader = DataLoader(dataset, batch_size=1)

    basepath = "./saves/"
    savepath = "./saves/winners/"
    progpath = "./saves/in-prog/"
    logfilename = basepath + "training.log"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    os.makedirs(os.path.dirname(progpath), exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename=logfilename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('Starting a  new run...')

    mdl = model.velvet()
    mdl.create(64, 64, 64, 500, 200, 0)
    attempts = 0
    wins = 0
    print(dataset[0])
    for j in range(0, 200000):
        attempts += 1
        tally = np.zeros(200)
        for k in range(0, 200):
            output = np.array(mdl.update(dataformat[j]["image"]))
            tally = tally + output
        guess = np.argmax(tally)
        answer = dataset[j]["label"]
        logging.info('{ "guess" : "' + str(guess) + '" }')
        logging.info('{ "answer"  : "' + str(answer) + '" }')
        if answer == guess:
            wins += 1
            logging.info('WIN! Wins so far: ' + str(wins))
            mdl.save(savepath)
            mdl.clear()
        else:
            mdl.permute(20)
    logging.info('Run ending...')
    logging.info('Total wins this run: ' + str(wins))
    logging.info('Total attempts this run: ' + str(attempts))
    mdl.save(progpath)
    return
    


