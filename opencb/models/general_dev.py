import numpy as np
from tensorplus import tensorplus as tp
import os

class general02():

    '''
    This version of the general purpose model will use the tensorplus library to speed things up a bit.
    Also, we'll be fixing a few bugs and adding functionality, particularly when it comes
    to neuron connections.
    '''
    #cuda device
    device = 0


    '''
    Step 1
    Take the input image and squishify it, then use it as starting values for the first layer
    
    step 2
    Propagate forward as normal, by multiplying each neuron's starting value with its weights and adding those values to the neurons in the next layer
    
    step 3
    multiply each of the weights between layer 1 and 2 by a value proportional to the sum of it's contribution to the values in layer 2.
    this is the emotion value that we used in the previous versions of the general model.
    the emotion value should then be modified by its personality value, which would mean that the emotion should also have the same process
    applied to it... in other words, the weights should look like:
    '''
    
    colors = 255
    size = 0
    image_size = 0
    image_map = 0
    maleability = 1.1
    range_high = 2.0
    range_low = 1/range_high
    
    num_controls = 0
    controls = 0
    control_thresholds_pos = 0
    control_thresholds_neg = 0
    
    num_sensations = 0
    sensations = 0
    
    neurons = 0
    thresholds_pos = 0
    thresholds_neg = 0
    signals_pos = 0
    signals_neg = 0
    emotion1 = 0
    emotion2 = 0
    emotion3 = 0
    emotion4 = 0
    emotion5 = 0
    emotion6 = 0
    emotion7 = 0
    emotion8 = 0
    personality1 = 0
    personality2 = 0
    personality3 = 0
    personality4 = 0
    personality5 = 0
    personality6 = 0
    personality7 = 0
    personality8 = 0
    personality9 = 0
    personality10 = 0
    personality11 = 0
    personality12 = 0
    personality13 = 0
    personality14 = 0
    personality15 = 0
    personality16 = 0
    connections1 = 0
    connections2 = 0
    connections3 = 0
    connections4 = 0
    connections5 = 0
    connections6 = 0
    connections7 = 0
    connections8 = 0
    
    __helper1 = 0
    __helper2 = 0
    __helper3 = 0
    
    propensity_pos = 0
    propensity_neg = 0
    
    firing_pos = 0
    resting_pos = 0
    firing_neg = 0
    resting_neg = 0
    
    fire_amt_pos = 0
    fire_amt_neg = 0
    
    fire_amt_mult_pos = 0
    fire_amt_mult_neg = 0
    
    propensity = 0
    
    def __init(self):
        return
    
    def __check_cuda(self):
        return
    
         
    def copy(self, model):
        '''
        Copy a model's parameters to a new model.
        
        :Parameters:
        model (general): the model to copy
        
        :Returns:
        none
        '''
        self.colors = model.colors
        self.size = model.size
        self.bounds = model.bounds
        self.range_high = model.range_high
        self.range_low = model.range_low
        self.num_controls = model.num_controls
        self.controls = model.controls
        self.control_thresholds_pos = model.control_thresholds_pos
        self.control_thresholds_neg = model.control_thresholds_neg
        self.num_sensations = model.num_sensations
        self.sensations = model.sensations
        
        self.neurons = model.neurons
        self.thresholds_pos = model.thresholds_pos
        self.thresholds_neg = model.thresholds_neg
        self.signals_pos = model.signals_pos
        self.signals_neg = model.signals_neg
        
        self.emotion1 = model.emotion1
        self.emotion2 = model.emotion2
        self.emotion3 = model.emotion3
        self.emotion4 = model.emotion4
        self.emotion5 = model.emotion5
        self.emotion6 = model.emotion6
        self.emotion7 = model.emotion7
        self.emotion8 = model.emotion8
        
        self.personality1 = model.personality1
        self.personality2 = model.personality2
        self.personality3 = model.personality3
        self.personality4 = model.personality4
        self.personality5 = model.personality5
        self.personality6 = model.personality6
        self.personality7 = model.personality7
        self.personality8 = model.personality8
        self.personality9 = model.personality9
        self.personality10 = model.personality10
        self.personality11 = model.personality11
        self.personality12 = model.personality12
        self.personality13 = model.personality13
        self.personality14 = model.personality14
        self.personality15 = model.personality15
        self.personality16 = model.personality16
        
        self.connections1 = model.connections1
        self.connections2 = model.connections2
        self.connections3 = model.connections3
        self.connections4 = model.connections4
        self.connections5 = model.connections5
        self.connections6 = model.connections6
        self.connections7 = model.connections7
        self.connections8 = model.connections8
        self.pos_propensity = model.pos_propensity
        self.neg_propensity = model.neg_propensity
        return

    
    def create(self, size, bounds, num_controls, num_sensations):
        self.__check_cuda()
        self.size = size
        self.bounds = bounds
        self.num_controls = num_controls
        self.num_sensations = num_sensations
        self.__new_helpers()
        self.__new_propensity()
        self.__new_image_map()
        self.__new_controls()
        self.__new_thresholds()
        self.__new_personality()
        self.__new_sensations()
        self.__new_connections()
        return
    
    def __new_range(self, r):
        self.range_low = np.arctan(2*r/np.pi) + 2
        self.range_high = 1 / self.range_low
        if (self.range_low > self.range_high):
            self.range_low, self.range_high = self.range_high, self.range_low
        return

    def __new_helpers(self):
        self.__helper1 = tp.create(self.size)
        self.__helper2 = tp.create(self.size)
        self.__helper3 = tp.create(self.size)
        return
    
    def __new_propensity(self):
       random_gen = np.random.default_rng(seed=None)
       random_gen.seed()
       self.propensity_pos = random_gen.integers(low=1, high=self.bounds, size=1)
       self.propensity_neg = random_gen.integers(low=-self.bounds, high=-1, size=1)
       return
   
    def __new_image_map(self):
        self.image_map = tp.create(self.size)
        tp.zeros(self.image_map)
        for i in range(0, self.image_size):
            tp.set(self.image_map, i, i)
    
    def __new_thresholds(self):
        random_gen = np.random.default_rng(seed=None)
        random_gen.seed()
        self.thresholds_pos = tp.create(self.size)
        self.thresholds_neg = tp.create(self.size)
        tp.zeros(self.thresholds_pos)
        tp.zeros(self.thresholds_neg)
        for i in range(0, self.size):
            tp.set(self.thresholds_pos, i, random_gen.integers(low=self.range_low, high=self.range_high,size=1))
            tp.set(self.thresholds_neg, i, random_gen.integers(low=self.range_low, high=self.range_high,size=1))
        return
    
    def __new_controls(self):
        self.controls = []
        for i in range(0, self.num_controls):
            self.controls.append((np.random.randint(low=1, high=self.size - 1)))
        return
    
    def __new_connections(self):
        random_gen = np.random.default_rng(seed=None)
        random_gen.seed()
        for i in range(0, self.size):
            tp.set(self.connections1, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections2, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections3, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections4, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections5, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections6, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections7, i, random_gen.integers(low=0, high=self.size, size=1))
            tp.set(self.connections8, i, random_gen.integers(low=0, high=self.size, size=1))
        return
    
    def __new_personality(self):
        self.neurons = tp.create(self.size)
        self.thresholds_pos = tp.create(self.size)
        self.thresholds_neg = tp.create(self.size)
        self.signals_pos = tp.create(self.size)
        self.signals_neg = tp.create(self.size)
        self.emotion1 = tp.create(self.size)
        self.emotion2 = tp.create(self.size)
        self.emotion3 = tp.create(self.size)
        self.emotion4 = tp.create(self.size)
        self.emotion5 = tp.create(self.size)
        self.emotion6 = tp.create(self.size)
        self.emotion7 = tp.create(self.size)
        self.emotion8 = tp.create(self.size)
        self.personality1 = tp.create(self.size)
        self.personality2 = tp.create(self.size)
        self.personality3 = tp.create(self.size)
        self.personality4 = tp.create(self.size)
        self.personality5 = tp.create(self.size)
        self.personality6 = tp.create(self.size)
        self.personality7 = tp.create(self.size)
        self.personality8 = tp.create(self.size)
        self.personality9 = tp.create(self.size)
        self.personality10 = tp.create(self.size)
        self.personality11 = tp.create(self.size)
        self.personality12 = tp.create(self.size)
        self.personality13 = tp.create(self.size)
        self.personality14 = tp.create(self.size)
        self.personality15 = tp.create(self.size)
        self.personality16 = tp.create(self.size)
        self.firing_pos = tp.create(self.size)
        self.resting_pos = tp.create(self.size)
        self.firing_neg = tp.create(self.size)
        self.resting_neg = tp.create(self.size)
        self.fire_amt_pos = tp.create(self.size)
        self.fire_amt_neg = tp.create(self.size)
        self.fire_amt_mult_pos = tp.create(self.size)
        self.fire_amt_mult_neg = tp.create(self.size)
        
        tp.zeros(self.neurons)
        tp.zeros(self.thresholds_pos)
        tp.zeros(self.thresholds_neg)
        tp.zeros(self.signals_pos)
        tp.zeros(self.signals_neg)
        tp.zeros(self.emotion1)
        tp.zeros(self.emotion2)
        tp.zeros(self.emotion3)
        tp.zeros(self.emotion4)
        tp.zeros(self.emotion5)
        tp.zeros(self.emotion6)
        tp.zeros(self.emotion7)
        tp.zeros(self.emotion8)
        tp.zeros(self.personality1)
        tp.zeros(self.personality2)
        tp.zeros(self.personality3)
        tp.zeros(self.personality4)
        tp.zeros(self.personality5)
        tp.zeros(self.personality6)
        tp.zeros(self.personality7)
        tp.zeros(self.personality8)
        tp.zeros(self.personality9)
        tp.zeros(self.personality10)
        tp.zeros(self.personality11)
        tp.zeros(self.personality12)
        tp.zeros(self.personality13)
        tp.zeros(self.personality14)
        tp.zeros(self.personality15)
        tp.zeros(self.personality16)
        tp.zeros(self.firing_pos)
        tp.zeros(self.resting_pos)
        tp.zeros(self.firing_neg)
        tp.zeros(self.resting_neg)
        tp.zeros(self.fire_amt_pos)
        tp.zeros(self.fire_amt_neg)
        tp.zeros(self.fire_amt_mult_pos)
        tp.zeros(self.fire_amt_mult_neg)
        
        random_gen = np.random.default_rng(seed=None)
        random_gen.seed()
        for i in range(0, self.size):
            tp.set(self.personality1, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality2, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality3, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality4, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality5, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality6, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality7, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality8, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality9, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality10, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality11, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality12, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality13, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality14, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
            tp.set(self.personality15, i, random_gen.integers(low=0, high=self.propensity_pos, size=1))
            tp.set(self.personality16, i, random_gen.integers(low=0, high=self.propensity_neg, size=1))
        return
    
    def __new_sensations(self):
        self.sensations = []
        for i in range(0, self.num_sensations):
            self.sensations.append((np.random.randint(low=1, high=self.size)))
        return

    def __pos_sensation(self, sense_num, amt):
        current = 0
        tp.get(self.neurons, self.sensations[sense_num], current)
        tp.set(self.neurons, self.sensations[sense_num], amt + current)
        return
    
    def __neg_sensation(self, sense_num, amt):
        current = 0
        tp.get(self.neurons, self.sensations[sense_num], current)
        tp.set(self.neurons, self.sensations[sense_num], amt + current)
        return
    
    def train(self, sense_num, amt, pos):
        if (pos):
            self.__pos_sensation(sense_num, amt)
        else:
            self.__neg_sensation(sense_num, amt)
        return
    
    
    def permute(self, degree, fraction):        
        '''
        Permute the model's personality by a certain degree.

        :Parameters:
        degree (int): positive integer which increases how much the permutation changes the model
        fraction (int): positive integer which lessens the degree of the permutation as it receives higher values

        :Returns:
        none

        :Comments: 
        You will absolutely need to trial and error with the degree to see what works best for your use case.
        This function will enable iterating on the personality traits of a model which has already proven useful.
        You'll want to use this to make small, incremental improvements to a model and then test it to see whether to move 
        forward with the changes or roll back to a previous version.
        
        If you want the model to change quickly, set the degree to a high number, and the fraction to 1.
        If you want the model to change slowly (and in most cases you will want this), set the degree to 1 and
        the fraction to higher numbers. The higher fraction goes, the slower the model will change with each iteration.

        Once a minimal working model has been found, this function will be what we primarily use to iterate on it.
        '''
        model = general2()
        model.copy(self)
        model.__new_thresholds()
        model.__new_propensity()
        model.__new_personality()
        degree = int(1 / fraction)
        tp.div_scalar(model.thresholds_pos, fraction, model.thresholds_pos)
        tp.div_scalar(model.thresholds_neg, fraction, model.thresholds_neg)
        tp.div_scalar(model.signals_pos, fraction, model.signals_pos)
        tp.div_scalar(model.signals_neg, fraction, model.signals_pos)
        tp.div_scalar(model.personality1, fraction, model.personality1)
        tp.div_scalar(model.personality1, fraction, model.personality2)
        tp.div_scalar(model.personality1, fraction, model.personality3)
        tp.div_scalar(model.personality1, fraction, model.personality4)
        tp.div_scalar(model.personality1, fraction, model.personality5)
        tp.div_scalar(model.personality1, fraction, model.personality6)
        tp.div_scalar(model.personality1, fraction, model.personality7)
        tp.div_scalar(model.personality1, fraction, model.personality8)
        tp.div_scalar(model.personality1, fraction, model.personality9)
        tp.div_scalar(model.personality1, fraction, model.personality10)
        tp.div_scalar(model.personality1, fraction, model.personality11)
        tp.div_scalar(model.personality1, fraction, model.personality12)
        tp.div_scalar(model.personality1, fraction, model.personality13)
        tp.div_scalar(model.personality1, fraction, model.personality14)
        tp.div_scalar(model.personality1, fraction, model.personality15)
        tp.div_scalar(model.personality1, fraction, model.personality16)
        for i in range(0, degree):
            tp.add(self.thresholds_pos, model.thresholds_pos, self.thresholds_pos)
            tp.add(self.thresholds_neg, model.thresholds_neg, self.thresholds_neg)
            tp.add(self.signals_pos, model.signals_pos, self.signals_pos)
            tp.add(self.signals_neg, model.signals_neg, self.signals_neg)
            tp.add(self.personality1, model.personality1, self.personality1)
            tp.add(self.personality2, model.personality2, self.personality2)
            tp.add(self.personality3, model.personality3, self.personality3)
            tp.add(self.personality4, model.personality4, self.personality4)
            tp.add(self.personality5, model.personality5, self.personality5)
            tp.add(self.personality6, model.personality6, self.personality6)
            tp.add(self.personality7, model.personality7, self.personality7)
            tp.add(self.personality8, model.personality8, self.personality8)
            tp.add(self.personality9, model.personality9, self.personality9)
            tp.add(self.personality10, model.personality10, self.personality10)
            tp.add(self.personality11, model.personality11, self.personality11)
            tp.add(self.personality12, model.personality12, self.personality12)
            tp.add(self.personality13, model.personality13, self.personality13)
            tp.add(self.personality14, model.personality14, self.personality14)
            tp.add(self.personality15, model.personality15, self.personality15)
            tp.add(self.personality16, model.personality16, self.personality16)
        degree+=1
        tp.div_scalar(self.thresholds_pos, degree, self.thresholds_pos)
        tp.div_scalar(self.thresholds_neg, degree, self.thresholds_neg)
        tp.div_scalar(self.signals_pos, degree, self.signals_pos)
        tp.div_scalar(self.signals_neg, degree, self.signals_neg)
        tp.div_scalar(self.personality1, degree, self.personality1)
        tp.div_scalar(self.personality1, degree, self.personality2)
        tp.div_scalar(self.personality1, degree, self.personality3)
        tp.div_scalar(self.personality1, degree, self.personality4)
        tp.div_scalar(self.personality1, degree, self.personality5)
        tp.div_scalar(self.personality1, degree, self.personality6)
        tp.div_scalar(self.personality1, degree, self.personality7)
        tp.div_scalar(self.personality1, degree, self.personality8)
        tp.div_scalar(self.personality1, degree, self.personality9)
        tp.div_scalar(self.personality1, degree, self.personality10)
        tp.div_scalar(self.personality1, degree, self.personality11)
        tp.div_scalar(self.personality1, degree, self.personality12)
        tp.div_scalar(self.personality1, degree, self.personality13)
        tp.div_scalar(self.personality1, degree, self.personality14)
        tp.div_scalar(self.personality1, degree, self.personality15)
        tp.div_scalar(self.personality1, degree, self.personality16)
        return
    
    def __convert_nparray(self, nparray):
        dims = nparray.shape
        size = dims[0] * dims[1]
        tensor = tp.create(size)
        for i in range(0, dims[0]):
            for j in range(0, dims[1]):
                tp.set(tensor, (i * dims[1]) + j, nparray[i, j])
        return tensor

    '''
    Accepts a numpy array that contains an input image
    '''
    def update(self, input_image):
        
        # reset these
        tp.zeros(self.__helper1)
        tp.zeros(self.__helper2)
        tp.zeros(self.__helper3)
        
        # prepare the input image
        input_tensor = self.__convert_nparray(input_image)
        tp.vector_resize(input_tensor, self.image_map, self.__helper1)
        # helper1 carries the resized image over to be added to the neuron layer
        tp.add(self.neurons, self.__helper1, self.neurons)
        
        # handle the firing neurons that exceeded their positive thresholds
        # helper1 now represents which neurons fired positively
        tp.greater(self.neurons, self.thresholds_pos, self.__helper1)
        
        # update the positive thresholds, signals, and emotions based on which neurons fired
        # here helper2 and helper3 just carry values between function calls
        # emotion 1 
        tp.mul(self.emotion1, self.__helper1, self.__helper2)
        tp.add(self.thresholds_pos, self.__helper2, self.thresholds_pos)
        tp.mul(self.personality1, self.__helper2, self.__helper3)
        tp.add(self.emotion1, self.__helper3, self.emotion1)
        tp.negate(self.__helper2)
        tp.mul(self.personality2, self.__helper2, self.__helper3)
        tp.add(self.emotion1, self.__helper3, self.emotion1)
        
        # emotion 5
        tp.mul(self.emotion5, self.__helper1, self.__helper2)
        tp.add(self.signals_pos, self.__helper2, self.signals_pos)
        tp.mul(self.personality9, self.__helper2, self.__helper3)
        tp.add(self.emotion5, self.__helper3, self.emotion5)
        tp.negate(self.__helper2)
        tp.mul(self.personality10, self.__helper2, self.__helper3)
        tp.add(self.emotion5, self.__helper3, self.emotion5)
        
        # emotion 2
        tp.negate(self.__helper1)
        tp.mul(self.emotion2, self.__helper1, self.__helper2)
        tp.add(self.thresholds_pos, self.__helper2, self.thresholds_pos)
        tp.mul(self.personality3, self.__helper2, self.__helper3)
        tp.add(self.emotion2, self.__helper3, self.emotion2)
        tp.negate(self.__helper2)
        tp.mul(self.personality4, self.__helper2, self.__helper3)
        tp.add(self.emotion2, self.__helper3, self.emotion2)
        
        # emotion 6
        tp.mul(self.emotion6, self.__helper1, self.__helper2)
        tp.add(self.signals_pos, self.__helper2, self.thresholds_pos)
        tp.mul(self.personality11, self.__helper2, self.__helper3)
        tp.add(self.emotion6, self.__helper3, self.emotion6)
        tp.negate(self.__helper2)
        tp.mul(self.personality12, self.__helper2, self.__helper3)
        tp.add(self.emotion6, self.__helper3, self.emotion6)
        
        # now have the neurons actually fire
        # this means they'll send their signals to each of their connected neurons
        tp.mul(self.signals_pos, self.__helper1, self.__helper1)
        tp.vector_add(self.neurons, self.__helper1, self.connections1, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections2, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections3, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections4, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections5, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections6, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections7, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections8, self.__helper2)
        tp.copy(self.__helper2, self.neurons)
        
        # now we repeat that process for the negative neurons
        tp.lesser(self.neurons, self.thresholds_neg, self.__helper1)
        
        # update the positive thresholds, signals, and emotions based on which neurons fired
        # here helper2 and helper3 just carry values between function calls
        # emotion 3
        tp.mul(self.emotion3, self.__helper1, self.__helper2)
        tp.add(self.thresholds_pos, self.__helper2, self.thresholds_pos)
        tp.mul(self.personality5, self.__helper2, self.__helper3)
        tp.add(self.emotion3, self.__helper3, self.emotion3)
        tp.negate(self.__helper2)
        tp.mul(self.personality6, self.__helper2, self.__helper3)
        tp.add(self.emotion3, self.__helper3, self.emotion3)
        
        # emotion 7
        tp.mul(self.emotion7, self.__helper1, self.__helper2)
        tp.add(self.signals_pos, self.__helper2, self.signals_pos)
        tp.mul(self.personality13, self.__helper2, self.__helper3)
        tp.add(self.emotion7, self.__helper3, self.emotion7)
        tp.negate(self.__helper2)
        tp.mul(self.personality14, self.__helper2, self.__helper3)
        tp.add(self.emotion7, self.__helper3, self.emotion7)
        
        # emotion 4
        tp.negate(self.__helper1)
        tp.mul(self.emotion4, self.__helper1, self.__helper2)
        tp.add(self.thresholds_pos, self.__helper2, self.thresholds_pos)
        tp.mul(self.personality7, self.__helper2, self.__helper3)
        tp.add(self.emotion4, self.__helper3, self.emotion4)
        tp.negate(self.__helper2)
        tp.mul(self.personality8, self.__helper2, self.__helper3)
        tp.add(self.emotion4, self.__helper3, self.emotion2)
        
        # emotion 8
        tp.mul(self.emotion8, self.__helper1, self.__helper2)
        tp.add(self.signals_pos, self.__helper2, self.thresholds_pos)
        tp.mul(self.personality15, self.__helper2, self.__helper3)
        tp.add(self.emotion8, self.__helper3, self.emotion8)
        tp.negate(self.__helper2)
        tp.mul(self.personality16, self.__helper2, self.__helper3)
        tp.add(self.emotion8, self.__helper3, self.emotion8) 
        
        # if a control threshold is met at this stage, handle that
        take_action = []
        for i in range(0, self.num_controls):
            if (self.neurons[self.controls[i]] > self.control_thresholds_pos[i]):
                take_action.append(True)
                tp.set(self.neurons, self.controls[i], self.control_thresholds_pos[i])
            else:
                if (self.neurons[self.controls[i] < self.control_thresholds_neg[i]]):
                    take_action.append(False)
                else:
                    take_action.append(-1)
        
        return take_action