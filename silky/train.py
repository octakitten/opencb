import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from . import model
import os
import random

# this isnt used currently. might deleted later...
def time_chamber():
    print("It's so over...")
    percent = 0.0
    while percent < .95:
        train("",  "")
        percent = test("", "")
    print("We're so back.")

def train(repo, path):

    # set up the dataset so it can be used on the gpu
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if repo == "" : repo = "Maysee/tiny-imagenet"
    dataset = datasets.load_dataset(repo, split="train")
    dataformat = dataset.with_format("torch", device=gpu)
    #batchsize = 8
    #dataloader = DataLoader(dataformat, batch_size=batchsize)

    # set up the save path and event logging
    if path == "":
        path = os.getcwd() + "/default/"
    basepath = path
    savepath = path + "winners"
    progpath = path + "in-prog"
    logfilename = basepath + "training.log"
    if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
    if not os.path.exists(progpath): os.makedirs(progpath, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename=logfilename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('Starting a  new run...')

    # initialize the model if we need to, but default to loading it in
    mdl = model.velvet()
    try:
        os.path.exists(savepath + "/width.npy")
        mdl.load(savepath)
    except:
        try:
            os.path.exists(progpath + "/width.npy")
            mdl.load(progpath)
        except:
            mdl.create(64, 64, 200, 500, 200, 0)
    attempts = 0
    wins = 0
    tolerance = 20

    # set up tools we need to randomize the data selection process
    len_dataset = len(dataformat)
    numbers_to_use = list(range(0, len_dataset))

    # loop over the dataset pseudo randomly
    for i in range(0, len_dataset):
        j = random.randint(0, len(numbers_to_use))
        n = numbers_to_use.pop(j)

        attempts += 1

        # the value here for exposure is important actually
        # the "exposure time" is the time that the model is given 
        # to process and understand each data element.
        # theoretically, the model should need at least a certain amount of exposure
        # time in order to make accurate predictions. but it consumes resources the 
        # longer the exposure time runs. this is something you'll have to
        # figure out a balance for as you work with training models
        exposure_time = 400
        tally = np.zeros(200)
        for k in range(0, exposure_time):
            output = np.array(mdl.update(dataformat[n]["image"]))
            tally = tally + output

        # see how the model did an log it.
        guess = np.argmax(tally)
        answer = dataformat[n]["label"].item()
        logging.info('{ "item# : "' + str(n) + '" }')
        logging.info('{ "guess" : "' + str(guess) + '" }')
        logging.info('{ "answer"  : "' + str(answer) + '" }')

        # when the model is right, we reward it by letting it survive intact, so to speak
        # we dont change parameters when the model wins, only when it loses
        # also, the tolerance here is important
        # see, the amount that the model changes when it loses can be fine
        # tuned with this tolerance number. if you want to change it, you can.
        # in here, its designed to change the model "less" when its winning a lot,
        # and "more" when its losing a lot. if the model doesnt get anything right
        # then it would seem like we have to change a lot of things about it, yea?
        if answer == guess:
            wins += 1
            logging.info('WIN! Wins so far: ' + str(wins))
            mdl.save(savepath)
            tolerance += 1
        else:
            mdl.permute(tolerance)
            if tolerance > 2 : tolerance -= 1
    logging.info('Run ending...')
    logging.info('Total wins this run: ' + str(wins))
    logging.info('Total attempts this run: ' + str(attempts))

    # make sure we save our in-progress models too.
    # sometimes these can be "better" than our last winning model, depending on circumstances
    mdl.save(progpath)
    return
    
def test(repo, path):
    # set up the dataset so it can be used on the gpu
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if repo == "" : repo = "Maysee/tiny-imagenet"
    dataset = datasets.load_dataset(repo, split="test")
    dataformat = dataset.with_format("torch", device=gpu)
    #batchsize = 8
    #dataloader = DataLoader(dataformat, batch_size=batchsize)

    # set up the save path and event logging
    if path == "":
        path = os.getcwd() + "/default/"
    basepath = path
    savepath = path + "winners"
    progpath = path + "in-prog"
    logfilename = basepath + "testing.log"
    logging.basicConfig(level=logging.DEBUG, filename=logfilename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('Starting a  new run...')

    # initialize the model if we need to, but default to loading it in
    mdl = model.velvet()
    try:
        os.path.exists(savepath + "/width.npy")
        mdl.load(savepath)
    except:
        try:
            os.path.exists(progpath + "/width.npy")
            mdl.load(progpath)
        except:
            print("Unable to find a model to load!")
    attempts = 0
    wins = 0

    # set up tools we need to randomize the data selection process
    len_dataset = len(dataformat)
    numbers_to_use = list(range(0, len_dataset))

    # loop over the dataset pseudo randomly
    for i in range(0, len_dataset):
        j = random.randint(0, len(numbers_to_use))
        n = numbers_to_use.pop(j)

        # the value here for exposure is important actually
        # the "exposure time" is the time that the model is given 
        # to process and understand each data element.
        # theoretically, the model should need at least a certain amount of exposure
        # time in order to make accurate predictions. but it consumes resources the 
        # longer the exposure time runs. this is something you'll have to
        # figure out a balance for as you work with training models
        attempts += 1
        exposure_time = 400
        tally = np.zeros(200)
        for k in range(0, exposure_time):
            output = np.array(mdl.update(dataformat[n]["image"]))
            tally = tally + output

        # see how the model did an log it.
        guess = np.argmax(tally)
        answer = dataformat[n]["label"].item()
        logging.info('{ "item# : "' + str(n) + '" }')
        logging.info('{ "guess" : "' + str(guess) + '" }')
        logging.info('{ "answer"  : "' + str(answer) + '" }')

        if answer == guess:
            wins += 1
            logging.info('WIN! Wins so far: ' + str(wins))
    logging.info('Run ending...')
    logging.info('Total wins this run: ' + str(wins))
    logging.info('Total attempts this run: ' + str(attempts))

    return wins/attempts


