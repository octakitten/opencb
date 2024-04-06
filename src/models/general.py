import torch
import numpy as np

class general():
    
    # records for how fast the model completes the game find_food_02
    min_dx = -1
    min_dy = -1

    # dimensions of the neural space
    width = 0
    height = 0
    depth = 0
    range = 255
    
    # how many control neurons there are
    num_controls = 0
    # array of those controls
    controls = []
    # array of those controls' positive and negative firing thresholds
    thresholds_pos = []
    thresholds_neg = []

    # how many sensation neurons there are
    num_sensations = 0
    # array of those neurons
    sensations = []

    
    # neuron layer
    layer0 = 0
    
    # threshold layers
    layer1 = 0
    layer2 = 0
    
    # neuron firing multipliers
    layer3 = 0
    layer4 = 0
    
    # threshhold emotion layers
    emotion1 = 0
    emotion2 = 0
    emotion3 = 0
    emotion4 = 0
    
    # multiplier emotion layers
    emotion5 = 0
    emotion6 = 0
    emotion7 = 0
    emotion8 = 0
    
    # threshold personality layers
    personality1 = 0
    personality2 = 0
    personality3 = 0
    personality4 = 0
    personality5 = 0
    personality6 = 0
    personality7 = 0
    personality8 = 0
    
    # multiplier personality layers
    personality9 = 0
    personality10 = 0
    personality11 = 0
    personality12 = 0
    personality13 = 0
    personality14 = 0
    personality15 = 0
    personality16 = 0
    
    
    # range of propensity to fire for personality layers
    pos_propensity = 0
    neg_propensity = 0
    
    positive_firing = 0
    positive_resting = 0
    negative_firing = 0
    negative_resting = 0

    # keep track of the values of the firing neurons
    pos_fire_amt = 0
    neg_fire_amt = 0
    
    # keep track of the values of the firing neurons when multiplied by layers 3 and 4
    pos_fire_amt_mult = 0
    neg_fire_amt_mult = 0


    def __init__(self):
        return
    
    def save(self, path):
        np.save(path, self.controls)
        np.save(path, self.thresholds_pos)
        np.save(path, self.thresholds_neg)
        torch.save(self.layer0, path)
        torch.save(self.layer1, path)
        torch.save(self.layer2, path)
        torch.save(self.emotion1, path)
        torch.save(self.emotion2, path)
        torch.save(self.emotion3, path)
        torch.save(self.emotion4, path)
        torch.save(self.personality1, path)
        torch.save(self.personality2, path)
        torch.save(self.personality3, path)
        torch.save(self.personality4, path)
        torch.save(self.personality5, path)
        torch.save(self.personality6, path)
        torch.save(self.personality7, path)
        torch.save(self.personality8, path)
        return
    
    '''
    Copy a model's parameters to a new model.
    
    Parameters:
    model (general): the model to copy
    
    Returns:
    none
    '''
    def copy(self, model):
        self.width = model.width
        self.height = model.height
        self.depth = model.depth
        self.num_controls = model.num_controls
        self.controls = model.controls
        self.thresholds_pos = model.thresholds_pos
        self.thresholds_neg = model.thresholds_neg
        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.emotion1 = model.emotion1
        self.emotion2 = model.emotion2
        self.emotion3 = model.emotion3
        self.emotion4 = model.emotion4
        self.personality1 = model.personality1
        self.personality2 = model.personality2
        self.personality3 = model.personality3
        self.personality4 = model.personality4
        self.personality5 = model.personality5
        self.personality6 = model.personality6
        self.personality7 = model.personality7
        self.personality8 = model.personality8
        self.pos_propensity = model.pos_propensity
        self.neg_propensity = model.neg_propensity
        return
    
    '''
    Create a new model with the given dimensions and number of controls.
    
    Parameters: 
    w (int): width of input images in pixels
    h (int): height of input images in pixels
    d (int): depth of the neural space
    num_controls (int): number of controls
    
    Returns:
    none
    
    This function creates a new, randomly initialized model with the dimensions and number of controls given.
    Importantly, this model will only be able to accept images of the specified width and height.
    The depth of the model determines its complexity. With more depth, the runtime and memory usage
    also increase dramatically. The number of controls determines what outputs the model can have. If you want it to 
    perform a certain task that requires, for instance, controlling 4 seperate keyboard keypresses, 
    then you would want a model with 4 controls.
    '''
    def create(self, w, h, d, r, num_controls, num_sensations):
        self.width = w
        self.height = h
        self.depth = d
        self.range = r
        self.num_controls = num_controls
        self.num_sensations = num_sensations
        self.__new_controls()
        self.__new_thresholds()
        self.__new_propensity()
        self.__new_personality()
        self.__new_sensations()
        return
    
    def __new_thresholds(self):
        self.thresholds_pos = []
        self.thresholds_neg = []
        threshhold_max = np.random.uniform(low=1, high=self.range)
        for i in range(0, self.num_controls):
            self.thresholds_pos.append(np.random.uniform(low=1, high=threshhold_max))
            self.thresholds_neg.append(np.random.uniform(low=1, high=threshhold_max))
        return
    
    def __new_controls(self):
        self.controls = []
        for i in range(0, self.num_controls):
            self.controls.append((np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), np.random.randint(low=1, high=self.depth)))
        return
    
    def __new_personality(self):
        
        # neuron layer
        self.layer0 = torch.zeros((self.width, self.height, self.depth))
        # threshold layers
        self.layer1 = torch.zeros((self.width, self.height, self.depth))
        self.layer2 = torch.zeros((self.width, self.height, self.depth))
        # emotion layers
        self.emotion1 = torch.zeros((self.width, self.height, self.depth))
        self.emotion2 = torch.zeros((self.width, self.height, self.depth))
        self.emotion3 = torch.zeros((self.width, self.height, self.depth))
        self.emotion4 = torch.zeros((self.width, self.height, self.depth))
        # personality layers
        # positive thresh firing is used
        self.personality1 = torch.zeros((self.width, self.height, self.depth))
        # positive thresh firing is unused
        self.personality2 = torch.zeros((self.width, self.height, self.depth))
        # positive thresh resting is used
        self.personality3 = torch.zeros((self.width, self.height, self.depth))
        # positive thresh resting is unused
        self.personality4 = torch.zeros((self.width, self.height, self.depth))
        # negative thresh firing is used
        self.personality5 = torch.zeros((self.width, self.height, self.depth))
        # negative thresh firing is unused
        self.personality6 = torch.zeros((self.width, self.height, self.depth))
        # negative thresh resting is used
        self.personality7 = torch.zeros((self.width, self.height, self.depth))
        # negative thresh resting is unused
        self.personality8 = torch.zeros((self.width, self.height, self.depth))
        self.personality9 = torch.zeros((self.width, self.height, self.depth))
        self.personality10 = torch.zeros((self.width, self.height, self.depth))
        self.personality11 = torch.zeros((self.width, self.height, self.depth))
        self.personality12 = torch.zeros((self.width, self.height, self.depth))
        self.personality13 = torch.zeros((self.width, self.height, self.depth))
        self.personality14 = torch.zeros((self.width, self.height, self.depth))
        self.personality15 = torch.zeros((self.width, self.height, self.depth))
        self.personality16 = torch.zeros((self.width, self.height, self.depth))
        
        self.positive_firing = torch.zeros((self.width, self.height, self.depth))
        self.positive_resting = torch.zeros((self.width, self.height, self.depth))
        self.negative_firing = torch.zeros((self.width, self.height, self.depth))
        self.negative_resting = torch.zeros((self.width, self.height, self.depth))
        self.pos_fire_amt = torch.zeros((self.width, self.height, self.depth))
        self.neg_fire_amt = torch.zeros((self.width, self.height, self.depth))
        
        # personality layers
        # positive thresh firing is used
        random_gen = torch.Generator()
        random_gen.seed()
        self.personality1 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # positive thresh firing is unused
        self.personality2 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # positive thresh resting is used
        self.personality3 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # positive thresh resting is unused
        self.personality4 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # negative thresh firing is used
        self.personality5 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # negative thresh firing is unused
        self.personality6 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # negative thresh resting is used
        self.personality7 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        random_gen.seed()
        # negative thresh resting is unused
        self.personality8 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        
        self.personality9 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality10 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality11 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality12 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality13 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality14 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality15 = torch.multiply(self.pos_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        self.personality16 = torch.multiply(self.neg_propensity, torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32))
        
        return
    
    def __new_propensity(self):
        self.pos_propensity = np.random.uniform(low=1, high=self.range)
        self.neg_propensity = -np.random.uniform(low=1, high=self.range)
        return

    def __new_sensations(self):
        self.sensations = []
        for i in range(0, self.num_sensations):
            self.sensations.append((np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), np.random.randint(low=1, high=self.depth)))
        return

    def __pos_sensation(self, sense_num, amt):
        torch.add(self.layer0[self.sensations[sense_num]], amt, out=self.layer0[self.sensations[sense_num]])
        return

    def __neg_sensation(self, sense_num, amt):
        torch.subtract(self.layer0[self.sensations[sense_num]], amt, out=self.layer0[self.sensations[sense_num]])
        return
    '''
    Train the model by giving it feedback on its actions.

    Parameters:
    sense_num (int): the index of the sensation neuron to train
    amt (float): the amount to train the sensation neuron by
    pos (bool): whether the sensation is positive or negative

    Returns:
    none

    Comments: Call this function whenever the model either does something right or makes a mistake.
    set pos to True if the sensation is positive, and False if the sensation is negative.
    You'll need to set conditions in your game that call this function automatically while it's playing.
    This function is also intended to be used later on in the training process, when the model is
    being used by a user on real world tasks and needs feedback.
    '''
    def train(self, sense_num, amt, pos):
        if (pos):
            self.__pos_sensation(sense_num, amt)
        else:
            self.__neg_sensation(sense_num, amt)
        return
    '''
    Permute the model's personality by a certain degree.

    Parameters:
    degree (int): positive integer which increases how much the permutation changes the model
    fraction (int): positive integer which lessens the degree of the permutation as it receives higher values

    Returns:
    none

    Comments: You will absolutely need to trial and error with the degree to see what works best for your use case.
    This function will enable iterating on the personality traits of a model which has already proven useful.
    You'll want to use this to make small, incremental improvements to a model and then test it to see whether to move 
    forward with the changes or roll back to a previous version.
    
    If you want the model to change quickly, set the degree to a high number, and the fraction to 1.
    If you want the model to change slowly (and in most cases you will want this), set the degree to 1 and
    the fraction to higher numbers. The higher fraction goes, the slower the model will change with each iteration.

    Once a minimal working model has been found, this function will be what we primarily use to iterate on it.
    '''
    def permute(self, degree, fraction):
        model = general()
        model.copy(self)
        model.__new_thresholds()
        model.__new_propensity()
        model.__new_personality()
        np.divide(model.thresholds_pos, fraction, out=model.thresholds_pos)
        np.divide(model.thresholds_neg, fraction, out=model.thresholds_neg)
        torch.divide(model.personality1, fraction, out=model.personality1)
        torch.divide(model.personality2, fraction, out=model.personality2)
        torch.divide(model.personality3, fraction, out=model.personality3)
        torch.divide(model.personality4, fraction, out=model.personality4)
        torch.divide(model.personality5, fraction, out=model.personality5)
        torch.divide(model.personality6, fraction, out=model.personality6)
        torch.divide(model.personality7, fraction, out=model.personality7)
        torch.divide(model.personality8, fraction, out=model.personality8)
        torch.divide(model.personality9, fraction, out=model.personality9)
        torch.divide(model.personality10, fraction, out=model.personality10)
        torch.divide(model.personality11, fraction, out=model.personality11)
        torch.divide(model.personality12, fraction, out=model.personality12)
        torch.divide(model.personality13, fraction, out=model.personality13)
        torch.divide(model.personality14, fraction, out=model.personality14)
        torch.divide(model.personality15, fraction, out=model.personality15)
        torch.divide(model.personality16, fraction, out=model.personality16)
        for i in range(0, degree):
            np.add(self.thresholds_pos, model.thresholds_pos, out=self.thresholds_pos)
            np.add(self.thresholds_neg, model.thresholds_neg, out=self.thresholds_neg)
            torch.add(self.personality1, model.personality1, out=self.personality1)
            torch.add(self.personality2, model.personality2, out=self.personality2)
            torch.add(self.personality3, model.personality3, out=self.personality3)
            torch.add(self.personality4, model.personality4, out=self.personality4)
            torch.add(self.personality5, model.personality5, out=self.personality5)
            torch.add(self.personality6, model.personality6, out=self.personality6)
            torch.add(self.personality7, model.personality7, out=self.personality7)
            torch.add(self.personality8, model.personality8, out=self.personality8)
            torch.add(self.personality9, model.personality9, out=self.personality9)
            torch.add(self.personality10, model.personality10, out=self.personality10)
            torch.add(self.personality11, model.personality11, out=self.personality11)
            torch.add(self.personality12, model.personality12, out=self.personality12)
            torch.add(self.personality13, model.personality13, out=self.personality13)
            torch.add(self.personality14, model.personality14, out=self.personality14)
            torch.add(self.personality15, model.personality15, out=self.personality15)
            torch.add(self.personality16, model.personality16, out=self.personality16)
        degree = degree / fraction
        np.divide(self.thresholds_pos, degree + 1, out=self.thresholds_pos)
        np.divide(self.thresholds_neg, degree + 1, out=self.thresholds_neg)
        torch.divide(self.personality1, degree + 1, out=self.personality1)
        torch.divide(self.personality2, degree + 1, out=self.personality2)
        torch.divide(self.personality3, degree + 1, out=self.personality3)
        torch.divide(self.personality4, degree + 1, out=self.personality4)
        torch.divide(self.personality5, degree + 1, out=self.personality5)
        torch.divide(self.personality6, degree + 1, out=self.personality6)
        torch.divide(self.personality7, degree + 1, out=self.personality7)
        torch.divide(self.personality8, degree + 1, out=self.personality8)
        torch.divide(self.personality9, degree + 1, out=self.personality9)
        torch.divide(self.personality10, degree + 1, out=self.personality10)
        torch.divide(self.personality11, degree + 1, out=self.personality11)
        torch.divide(self.personality12, degree + 1, out=self.personality12)
        torch.divide(self.personality13, degree + 1, out=self.personality13)
        torch.divide(self.personality14, degree + 1, out=self.personality14)
        torch.divide(self.personality15, degree + 1, out=self.personality15)
        torch.divide(self.personality16, degree + 1, out=self.personality16)
        
        return
    
    # save the model to a specified file
    def save(self, path):
        torch.save(self, path)
        return

    '''
    Main control function for the model.

    Parameters:
    input_image (tensor): the image to input into the model

    Returns:
    take_action (list): a list of booleans representing whether the controls should be activated or not

    Comments: This function is what makes the model 'act'. It takes an image as input and processes it by firing neurons.
    Usually a single image will not be enough to cause the model to take any action - you'll need to feed it a continuous
    stream of images that the model can react to and see its reactions change things in the image, as well. This style of 
    model doesn't work if it can't interact with its environment, so you'll need to have the model play a predefined game
    of your design or choosing, otherwise it won't do anything useful. 

    The game which you use or create should give the model a tensor image and then call this function to get the model's
    next action. The model will then return a list of booleans, each representing whether each control it has should be activated or not.
    It's up to you to make those controls do something in its environment.

    Provided in this library are some simple example games to get you started. You can also look at the testing scripts to see how to 
    implement them.
    '''
    def update(self, input_image):
        if (torch.is_tensor(input_image) == False):
            return -1
        # add in the input image
        torch.add(self.layer0[:, :, 1],  input_image, alpha=1, out=self.layer0[:, :, 1])

        #check which neurons are firing and which arent, do the stuff
        torch.greater(self.layer0, self.layer1, out=self.positive_firing)
        torch.less_equal(self.layer0, self.layer1, out=self.positive_resting)
        torch.greater(self.layer0, self.layer2, out=self.negative_firing)
        torch.less_equal(self.layer0, self.layer2, out=self.negative_resting)

        # keep track of the threshold values of the firing neurons
        torch.multiply(self.positive_firing, self.layer1, out=self.pos_fire_amt)
        torch.multiply(self.negative_firing, self.layer2, out=self.neg_fire_amt)
        
        # use the firing multipliers to change the output values of the firing neurons
        torch.multiply(self.pos_fire_amt, self.layer3, out=self.pos_fire_amt_mult)
        torch.multiply(self.neg_fire_amt, self.layer4, out=self.neg_fire_amt_mult)

        # apply the firing values to each of the near neighbors
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, 1, 0), alpha=1, out=self.layer0)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, -1, 0), alpha=1, out=self.layer0)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, 1, 1), alpha=1, out=self.layer0)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, -1, 1), alpha=1, out=self.layer0)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, 1, 2), alpha=1, out=self.layer0)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, -1, 2), alpha=1, out=self.layer0)
        torch.subtract(self.layer0, torch.roll(self.neg_fire_amt_mult, 1, 0), alpha=1, out=self.layer0)
        torch.subtract(self.layer0, torch.roll(self.neg_fire_amt_mult, -1, 0), alpha=1, out=self.layer0)
        torch.subtract(self.layer0, torch.roll(self.neg_fire_amt_mult, 1, 1), alpha=1, out=self.layer0)
        torch.subtract(self.layer0, torch.roll(self.neg_fire_amt_mult, -1, 1), alpha=1, out=self.layer0)
        torch.subtract(self.layer0, torch.roll(self.neg_fire_amt_mult, 1, 2), alpha=1, out=self.layer0)
        torch.subtract(self.layer0, torch.roll(self.neg_fire_amt_mult, -1, 2), alpha=1, out=self.layer0)
        
        # check the predefined output neurons to see if they're ready to fire
        # if they are, then return the action(s) to take
        take_action = []
        
        for i in range(0, self.num_controls):
            print(self.layer0[self.controls[i]].item())
            if (self.layer0[self.controls[i]].item() > self.thresholds_pos[i]):
                take_action.append(True)
                self.layer0[self.controls[i]] = self.layer0[self.controls[i]] - self.thresholds_pos[i]
            else:
                if (self.layer0[self.controls[i]].item() > self.thresholds_neg[i]):
                    take_action.append(False)
                else:
                    take_action.append(-1)
        
        # once a neuron has fired, its value needs to be lowered by the threshold amount
        torch.subtract(self.layer0, self.pos_fire_amt, alpha=1, out=self.layer0)
        torch.add(self.layer0, self.neg_fire_amt, alpha=1, out=self.layer0)

        # update the emotion (threshold) layers
        torch.add(self.layer1, torch.multiply(self.positive_firing, self.emotion1), alpha=1, out=self.layer1)
        torch.add(self.layer1, torch.multiply(self.positive_resting, self.emotion2), alpha=1, out=self.layer1)
        torch.add(self.layer2, torch.multiply(self.negative_firing, self.emotion3), alpha=1, out=self.layer2)
        torch.add(self.layer2, torch.multiply(self.negative_resting, self.emotion4), alpha=1, out=self.layer2)
        
        # update the emotion (multiplier) layers
        torch.add(self.layer3, torch.multiply(self.positive_firing, self.emotion5), alpha=1, out=self.layer3)
        torch.add(self.layer3, torch.multiply(self.positive_resting, self.emotion6), alpha=1, out=self.layer3)
        torch.add(self.layer4, torch.multiply(self.negative_firing, self.emotion7), alpha=1, out=self.layer4)
        torch.add(self.layer4, torch.multiply(self.negative_resting, self.emotion8), alpha=1, out=self.layer4)

        # figure out which emotions (thresholds) were used and which weren't
        # and then update them
        torch.add(torch.multiply(self.positive_firing, self.personality1), torch.multiply(self.positive_resting, self.personality3), alpha=1, out=self.emotion1)
        torch.add(torch.multiply(self.positive_resting, self.personality2), torch.multiply(self.positive_firing, self.personality4), alpha=1, out=self.emotion2)
        torch.add(torch.multiply(self.negative_firing, self.personality5), torch.multiply(self.negative_resting, self.personality7), alpha=1, out=self.emotion3)
        torch.add(torch.multiply(self.negative_resting, self.personality6), torch.multiply(self.negative_firing, self.personality8), alpha=1, out=self.emotion4)
        
        # do the same for the emotion multipliers
        torch.add(torch.multiply(self.positive_firing, self.personality9), torch.multiply(self.positive_resting, self.personality11), alpha=1, out=self.emotion5)
        torch.add(torch.multiply(self.positive_resting, self.personality10), torch.multiply(self.positive_firing, self.personality12), alpha=1, out=self.emotion6)
        torch.add(torch.multiply(self.negative_firing, self.personality13), torch.multiply(self.negative_resting, self.personality15), alpha=1, out=self.emotion7)
        torch.add(torch.multiply(self.negative_resting, self.personality14), torch.multiply(self.negative_firing, self.personality16), alpha=1, out=self.emotion8)
        
        #print(take_action)
        return take_action