import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from . import model
import os
import random

def time_chamber(options = None):
    print("Here we go...")
    if options == None:
        options = optionsobj("Maysee/tiny-imagenet", "", 64, 64, 200, 500, 200, 2)
    percent = 0.0
    while percent < .95:
        train(options)
        percent = test(optionsobj)
    print("Let's fucking GO!")

class optionsobj():
    '''
    This object is used to pass options to the training and testing functions.

    repo: the name of the dataset to use. This is the name of the dataset in the huggingface datasets library.
    path: the path to save the model and logs to. This is a string that should be a valid path on your system.
    height: the height of the input images. This is an integer.
    width: the width of the input images. This is an integer.
    depth: the depth of the input images. This is an integer.
    bounds: the number of bounds to use in the model. This is an integer.
    controls: the number of controls to use in the model. This is an integer.
    senses: the number of senses to use in the model. This is an integer.
    '''
    repo = None
    hftoken = None
    path = None
    height = None
    width = None
    depth = None
    bounds = None
    controls = None
    senses = None
    exposure = None
    def __init__(self, repo = None, hftoken = None, path = None, 
                 height = None, width = None, depth = None, 
                 bounds = None, controls = None, senses = None, exposure = None):
        self.repo = repo
        self.hftoken = hftoken
        self.path = path
        self.height = height
        self.width = width
        self.depth = depth
        self.bounds = bounds
        self.controls = controls
        self.senses = senses

def transforms(data, size):
    '''
    This function takes in a dataset object and applies some transformations to it. 
    This is useful if you want to resize the images or do some other kind of transformation
    to the data before training or testing a model.

    data: a dataset object from the huggingface datasets library.
    '''
    for i in range(0, len(data)):
        data[i]["image"] = torch.nn.functional.interpolate(data[i]["image"], size)
    return data

def train(options):
    '''
    This function trains a model on a dataset. You will need to decide on the parameters you want
    the model to have and pass them into this function with the options object. Thats kind of just for
    simplicity and readability's sake.

    options: an optionsobj object that contains the options for the training run.
    '''
    # some basic error checking
    if options.repo == "" : options.repo = "Maysee/tiny-imagenet"
    if options.repo == None: raise ValueError("No dataset provided!")
    if options.path == None: raise ValueError("No save path provided!")
    if options.height == None: raise ValueError("No height provided!")
    if options.width == None: raise ValueError("No width provided!")
    if options.depth == None: raise ValueError("No depth provided!")
    if options.bounds == None: raise ValueError("No bounds provided!")
    if options.controls == None: raise ValueError("No controls provided!")
    if options.senses == None: raise ValueError("No senses provided!")

    # set up the dataset so it can be used on the gpu
    # also resize the images to a height and width that matches the model's input stream
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if options.hftoken != None:
        dataset = datasets.load_dataset(options.repo, split="train", api_token=options.hftoken)
    else:
        dataset = datasets.load_dataset(options.repo, split="train")
    dataformat = dataset.with_format("torch", device=gpu)
    size = (options.height, options.width)
    dataformat = dataformat.map(transforms, dataformat, size, batched=True)

    # set up the save path and event logging
    if options.path == "":
        path = os.getcwd() + "/default/"
    basepath = options.path
    savepath = options.path + "winners"
    progpath = options.path + "in-prog"
    logfilename = basepath + "training.log"
    if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
    if not os.path.exists(progpath): os.makedirs(progpath, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename=logfilename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('Starting a  new run...')

    # initialize the model if we need to, but default to loading it in
    mdl = model.ferret()
    try:
        os.path.exists(savepath + "/width.npy")
        mdl.load(savepath)
    except:
        try:
            os.path.exists(progpath + "/width.npy")
            mdl.load(progpath)
        except:
            try:
                mdl.create(options.height, options.width, options.depth, options.bounds, options.controls, options.senses) 
            except:
                raise ValueError("Unable to create or load a model! Maybe try setting the model options with the options object.")
    attempts = 0
    wins = 0
    tolerance = 20
    permute_fraction = 20

    # set up tools we need to randomize the data selection process
    len_dataset = len(dataformat)
    numbers_to_use = list(range(0, len_dataset))
    last_win = 0

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
        if options.exposure == None: exposure_time = 400
        else: exposure_time = options.exposure
        tally = np.zeros(200)
        answer = dataformat[n]["label"].item()
        answerkey = np.zeros(200)
        for k in range(0, exposure_time):
            output = np.array(mdl.update(dataformat[n]["image"]))
            tally = tally + output
            answerkey[answer] += 1

        # see how the model did an log it.
        guess = np.argmax(tally)
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

        # also were adding a backprop function in on this step. not sure yet how well it will
        # work but, one thing we have to do is prepare an 'answer key' of which values are 
        # correct and which are not.

        if answer == guess:
            wins += 1
            logging.info('WIN! Wins so far: ' + str(wins))
            mdl.save(savepath)
            tolerance += 20
            last_win = 0
        else:
            if tolerance < 5:
                mdl.permute(1, permute_fraction)
                tolerance += 20
            last_win += 1
            cons = 0.1
            mdl.backprop(answerkey, tally, cons)
            mdl.permute(1, tolerance)
            if tolerance > 5: 
                tolerance -= 1
    logging.info('Run ending...')
    logging.info('Total wins this run: ' + str(wins))
    logging.info('Total attempts this run: ' + str(attempts))

    # make sure we save our in-progress models too.
    # sometimes these can be "better" than our last winning model, depending on circumstances
    mdl.save(progpath)
    return
    
def test(options):
    # set up the dataset so it can be used on the gpu
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = options.repo
    if repo == "" : options.repo = "Maysee/tiny-imagenet"
    if repo == None: repo = "Maysee/tiny-imagenet"
    dataset = datasets.load_dataset(options.repo, split="test")
    dataformat = dataset.with_format("torch", device=gpu)
    #batchsize = 8
    #dataloader = DataLoader(dataformat, batch_size=batchsize)

    # set up the save path and event logging
    path = options.path
    if options.path == None:
        path = os.getcwd() + "/default/"
    basepath = path
    savepath = path + "winners"
    progpath = path + "in-prog"
    logfilename = basepath + "testing.log"
    logging.basicConfig(level=logging.DEBUG, filename=logfilename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info('Starting a  new run...')

    # initialize the model if we need to, but default to loading it in
    mdl = model.ferret()
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


