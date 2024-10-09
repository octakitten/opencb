import numpy as np
from tensorplus import tensorplus as tp
import os

class general_dev():

    '''
    This version of the general purpose model will use the tensorplus library to speed things up a bit.
    Also, we'll be fixing a few bugs and adding functionality, particularly when it comes
    to neuron connections.
    '''
    #cuda device
    device = 0

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
    
         
    def copy(self, model: ocb.general_dev):
        '''
        Copy a model's data to a new model.
        
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
    
    def create(self, size: int, image_size: int, bounds: int, num_controls: int, num_sensations: int):
        '''
        Create the model, a potentially time consuming process.

        :Parameters:
        size (int): the number of neurons to use
        image_size (int): the size of the input image to use
        bounds (int): the limit of how large values in the personality traits can get
        num_controls (int): how many control neurons you want
        num_sensations (int): how many sensation neurons you want

        :Returns:
        none

        :Comments:
        If the cuda check fails, it throws an error. This function is what actually initializes the model.
        On init doesn't really do anything, and sometimes we want to create a blank model without
        going through the lengthy process of initializing all its values and saving them to the GPU.
        Use this to actually create the model before you start using it.

        '''
        self.__check_cuda()
        self.size = size
        self.image_size = image_size
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

    def __convert_tensor_to_nparray(tensor: tp.tensor) -> np.ndarray:
        arr = np.zeros(tensor.size)
        for i in range(0, tensor.size):
            arr[i] = tensor.get(i)
        tp.destroy(tensor)
        return arr

    def save(self, path: string):
        '''
        Save the model's data to disk.

        :Parameters:
        path (string): relative path from the location it's called at.

        :Returns:
        none

        :Comments:
        Saves the model's data to the harddisk for retrieval later. Wherever you ran a python file to call this function
        is the directory that this function will use to look for its path. There isn't any real searching involved here,
        so make sure the path you're looking for exists in that directory before you try to use this.
        '''
        if (not os.path.exists(path)):
            try:
                os.path.create(path)
            except:
                FileError("Error creating directory: unable to create directory.")
        np.save(path + 'size', self.size)
        np.save(path + 'image_size', self.image_size)
        np.save(path + 'bounds', self.bounds)
        np.save(path + 'neurons', __convert_tensor_to_nparray(self.neurons))
        np.save(path + 'thresholds_pos', __convert_tensor_to_nparray(self.thresholds_pos))
        np.save(path + 'thresholds_neg', __convert_tensor_to_nparray(self.thresholds_neg))
        np.save(path + 'signals_pos', __convert_tensor_to_nparray(self.signals_pos))
        np.save(path + 'signals_neg', __convert_tensor_to_nparray(self.signals_neg))
        np.save(path + 'emotions1', __convert_tensor_to_nparray(self.emotions1))
        np.save(path + 'emotions2', __convert_tensor_to_nparray(self.emotions2))
        np.save(path + 'emotions3', __convert_tensor_to_nparray(self.emotions3))
        np.save(path + 'emotions4', __convert_tensor_to_nparray(self.emotions4))
        np.save(path + 'emotions5', __convert_tensor_to_nparray(self.emotions5))
        np.save(path + 'emotions6', __convert_tensor_to_nparray(self.emotions6))
        np.save(path + 'emotions7', __convert_tensor_to_nparray(self.emotions7))
        np.save(path + 'emotions8', __convert_tensor_to_nparray(self.emotions8))
        np.save(path + 'personality1', __convert_tensor_to_nparray(self.personality1))
        np.save(path + 'personality2', __convert_tensor_to_nparray(self.personality2))
        np.save(path + 'personality3', __convert_tensor_to_nparray(self.personality3))
        np.save(path + 'personality4', __convert_tensor_to_nparray(self.personality4))
        np.save(path + 'personality5', __convert_tensor_to_nparray(self.personality5))
        np.save(path + 'personality6', __convert_tensor_to_nparray(self.personality6))
        np.save(path + 'personality7', __convert_tensor_to_nparray(self.personality7))
        np.save(path + 'personality8', __convert_tensor_to_nparray(self.personality8))
        np.save(path + 'personality8', __convert_tensor_to_nparray(self.personality9))
        np.save(path + 'personality10', __convert_tensor_to_nparray(self.personality10))
        np.save(path + 'personality11', __convert_tensor_to_nparray(self.personality11))
        np.save(path + 'personality12', __convert_tensor_to_nparray(self.personality12))
        np.save(path + 'personality13', __convert_tensor_to_nparray(self.personality13))
        np.save(path + 'personality14', __convert_tensor_to_nparray(self.personality14))
        np.save(path + 'personality15', __convert_tensor_to_nparray(self.personality15))
        np.save(path + 'personality16', __convert_tensor_to_nparray(self.personality16)) 
        np.save(path + 'connections1', __convert_tensor_to_nparray(self.connections1))
        np.save(path + 'connections2', __convert_tensor_to_nparray(self.connections2))
        np.save(path + 'connections3', __convert_tensor_to_nparray(self.connections3))
        np.save(path + 'connections4', __convert_tensor_to_nparray(self.connections4))
        np.save(path + 'connections5', __convert_tensor_to_nparray(self.connections5))
        np.save(path + 'connections6', __convert_tensor_to_nparray(self.connections6))
        np.save(path + 'connections7', __convert_tensor_to_nparray(self.connections7))
        np.save(path + 'connections8', __convert_tensor_to_nparray(self.connections8))
        np.save(path + 'num_controls', self.num_controls)
        np.save(path + 'controls', self.controls)
        np.save(path + 'control_thresholds_pos', self.control_thresholds_pos)
        np.save(path + 'control_thresholds_neg', self.control_thresholds_neg)
        np.save(path + 'num_sensations', self.num_sensations)
        np.save(path + 'sensations', self.sensations)
        np.save(path + 'propensity', self.propensity)
        np.save(path + 'propensity_pos', self.propensity_pos)
        np.save(path + 'propensity_neg', self.propensity_neg)
        self.__new_helpers()
        return
    
    def load(self, path: string):
        '''
        Load a model from storage.

        :Parameters:
        path (string): relative path from the location it's called from.

        :Returns:
        none

        :Comments:
        Currently we offload the functionality of saving data to the disk to NumPy. Eventually that will be fixed
        and included in tensorplus, but for now it just means that loading is slow. Also, be aware that 
        there's no real searching your system PATH variable for this directory. You just need to run your python program
        from the right directory in the first place.
        '''

        self.size = np.load(path + 'size')
        self.image_size = np.load(path + 'image_size')
        self.neurons = self.__convert_nparray(np.load(path + 'neurons'))                            
        self.thresholds_pos = self.__convert_nparray(np.load(path + 'thresholds_pos'))
        self.thresholds_neg = self.__convert_nparray(np.load(path + 'thresholds_neg'))
        self.signals_pos = self.__convert_nparray(np.load(path + 'signals_pos'))
        self.signals_neg = self.__convert_nparray(np.load(path + 'signals_neg'))
        self.emotions1 = self.__convert_nparray(np.load(path + 'emotions1'))
        self.emotions2 = self.__convert_nparray(np.load(path + 'emotions2'))
        self.emotions3 = self.__convert_nparray(np.load(path + 'emotions3'))
        self.emotions4 = self.__convert_nparray(np.load(path + 'emotions4'))
        self.emotions5 = self.__convert_nparray(np.load(path + 'emotions5'))
        self.emotions6 = self.__convert_nparray(np.load(path + 'emotions6'))
        self.emotions7 = self.__convert_nparray(np.load(path + 'emotions7'))
        self.emotions8 = self.__convert_nparray(np.load(path + 'emotions8'))
        self.personality1 = self.__convert_nparray(np.load(path + 'personality1'))
        self.personality2 = self.__convert_nparray(np.load(path + 'personality2'))
        self.personality3 = self.__convert_nparray(np.load(path + 'personality3'))
        self.personality4 = self.__convert_nparray(np.load(path + 'personality4'))
        self.personality5 = self.__convert_nparray(np.load(path + 'personality5'))
        self.personality6 = self.__convert_nparray(np.load(path + 'personality6'))
        self.personality7 = self.__convert_nparray(np.load(path + 'personality7'))
        self.personality8 = self.__convert_nparray(np.load(path + 'personality8'))
        self.personality9 = self.__convert_nparray(np.load(path + 'personality9'))
        self.personality10 = self.__convert_nparray(np.load(path + 'personality10'))
        self.personality11 = self.__convert_nparray(np.load(path + 'personality11'))
        self.personality12 = self.__convert_nparray(np.load(path + 'personality12'))
        self.personality13 = self.__convert_nparray(np.load(path + 'personality13'))
        self.personality14 = self.__convert_nparray(np.load(path + 'personality14'))
        self.personality15 = self.__convert_nparray(np.load(path + 'personality15'))
        self.personality16 = self.__convert_nparray(np.load(path + 'personality16'))
        self.connections1 = self.__convert_nparray(np.load(path + 'connections1'))
        self.connections2 = self.__convert_nparray(np.load(path + 'connections2'))
        self.connections3 = self.__convert_nparray(np.load(path + 'connections3'))
        self.connections4 = self.__convert_nparray(np.load(path + 'connections4'))
        self.connections5 = self.__convert_nparray(np.load(path + 'connections5'))
        self.connections6 = self.__convert_nparray(np.load(path + 'connections6'))
        self.connections7 = self.__convert_nparray(np.load(path + 'connections7'))
        self.connections8 = self.__convert_nparray(np.load(path + 'connections8'))
        self.num_controls = np.load(path + 'num_controls')
        self.controls = np.load(path + 'controls')
        self.control_thresholds_pos = np.load(path + 'control_thresholds_pos')
        self.control_thresholds_neg = np.load(path + 'control_thresholds_neg')
        self.num_sensations = np.load(path + 'num_sensations')
        self.sensations = np.load(path + 'num_sensations')

        return


    def __new_range(self, r: int):
        '''
        This function is currently not used.
        '''
        self.range_low = np.arctan(2*r/np.pi) + 2
        self.range_high = 1 / self.range_low
        if (self.range_low > self.range_high):
            self.range_low, self.range_high = self.range_high, self.range_low
        return

    def __new_helpers(self):
        '''
        Initializes the helper variables that we use in the update function to carry
        values between function calls.
        '''
        self.__helper1 = tp.create(self.size)
        self.__helper2 = tp.create(self.size)
        self.__helper3 = tp.create(self.size)
        return
    
    def __new_propensity(self):
        '''
        Creates the propensity values.
        :Parameters:
        none

        :Returns:
        none

        :Comments:
        The propensity is used to determine the potential sizes of the personality values. It's basically
        an upper and lower limit on how large values in the model will get.
        '''   
        random_gen = np.random.default_rng(seed=None)
        random_gen.seed()
        self.propensity_pos = random_gen.integers(low=1, high=self.bounds, size=1)
        self.propensity_neg = random_gen.integers(low=-self.bounds, high=-1, size=1)
        return

    def __new_image_map(self):
        '''
        Creates the image map.

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        The image map is what allows us to take an image smaller than the number of neurons and send the color values
        from that image to the neurons they're supposed to go to. The map will usually contain the numbers 1, 2, 3... n-2, n-1, n ,
        where n is the size of the image to be mapped. Those numbers will go in the first n indexes of the image map, the rest of the values will between
        simply 0. We later use this as a set of vectors for the image to the neurons, with the 0 values discarded.
        '''
        self.image_map = tp.create(self.size)
        tp.zeros(self.image_map)
        for i in range(0, self.image_size):
            tp.set(self.image_map, i, i)
    
    def __new_thresholds(self):
        '''
        Creates the thresholds for the neurons.

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        Contains values that correspond to the neurons and which signify the point at which the neuron will fire.
        When this value is exceeded either positively for the positive thresholds, or vice versa for the negatives,
        the neuron will fire and send its signal to each of its connected neurons (in the update() function).
        Here we initialize these threshold values to random values using a random generator from NumPy.
        '''
        random_gen = np.random.default_rng()
        self.thresholds_pos = tp.create(self.size)
        self.thresholds_neg = tp.create(self.size)
        tp.zeros(self.thresholds_pos)
        tp.zeros(self.thresholds_neg)
        for i in range(0, self.size):
            tp.set(self.thresholds_pos, i, random_gen.integers(low=self.range_low, high=self.range_high,size=1))
            tp.set(self.thresholds_neg, i, random_gen.integers(low=self.range_low, high=self.range_high,size=1))
        return
    
    def __new_controls(self):
        '''
        Creates the control neurons.

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        The control neurons are what allows the model to interact with its environment. Whenever the value at a control neuron
        exceeds its threshold, either positively or negatively, it will fire the neuron and take an action that has been
        programmatically assigned to it in a higher level codespace. Here we initialize them to random neurons.
        '''
        self.controls = []
        for i in range(0, self.num_controls):
            self.controls.append((np.random.randint(low=1, high=self.size - 1)))
        return
    
    def __new_connections(self):
        '''
        Creates the neuron connection vectors.

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        The connections are vectors that map the neurons to other neurons. Whenever a given neuron fires, it sends its signal value to
        each other neuron that it's connected to. Note that connections are one-way and not necessarily reversible - indeed, most of the time 
        neurons won't be reciprocally connected unless it happens at random.
        '''
        random_gen = np.random.default_rng()
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
        '''
        Creates a new personality for the model.

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        The personality is what allows the model to have both persistent memory and a predefined way it reacts to information.
        This will take some explaining:

        First of all, the personality values are static in the model. During operation they don't change. These are roughly analogous 
        to the model weights that are used in typical deep learning neural networks. We iterate on the personality values in order
        to create new models and test them against the ones we already have.

        To describe the whole process, when a neuron fires in the update function, it uses a threshold value and a signal value. 
        Whenever a threshold value or a signal value is used, that value gets modified 
        by a corresponding emotion value - we simply add the emotion value to the threshold or signal value that's 
        in that spot. The catch is that when a threshold value or signal value is NOT used, we ALSO modify that value by a Different emotion value. 
        Here, it looks like this:

        -----

        if a neuron fires positively, threshold_pos and signal_pos are used, while threshold_neg and signal_neg are NOT used
        if a neuron fires negatively, threshold_pos and signal_pos are used, while threshold_pos and signal_pos are NOT used 
        if a neuron does not fire at all, neither the positive or negative thresholds and signals are used 
        
        Then:
        if threshold_pos is used, emotion1 is added to it
        if threshold_pos is NOT used, emotion2 is added to it 

        if threshold_neg is used, emotion3 is added to it 
        if threshold_neg is NOT used, emotion4 is added to it 

        if signal_pos is used, emotion5 is added to it 
        if signal_pos is NOT used, emotion6 is added to it 

        if signal_neg is used, emotion7 is added to it 
        if signal_neg is NOT used, emotion8 is added to it 

        -----

        To continue the explanation, this isn't the end of what we do when a neuron fires. We also repeat this process in a similar
        manner for the emotion layers. In much the same way, if an emotion is added to its corresponding threshold or signal, then 
        we add a certain personality value to it, and if it is NOT added to its corresponding threshold or signal, then we add a 
        different personality value to it. It'll look like this:

        -----

        if emotion1 is used, personality1 is added to it 
        if emotion1 is NOT used, personality2 is added to it 

        if emotion2 is used, personality3 is added to it 
        if emotion2 is NOT used, personality4 is added to it 

        if emotion3 is used, personality5 is added to it 
        if emotion3 is NOT used, personality6 is added to it 

        if emotion4 is used, personality7 is added to it 
        if emotion4 is NOT used, personality8 is added to it 

        if emotion5 is used, personality9 is added to it 
        if emotion5 is NOT used, personality10 is added to it 

        if emotion6 is used, personality11 is added to it 
        if emotion6 is NOT used, personality12 is added to it 

        if emotion7 is used, personality13 is added to it 
        if emotion7 is NOT used, personality14 is added to it 

        if emotion8 is used, personality15 is added to it 
        if emotion8 is NOT used, personality16 is added to it 

        -----

        This elaborate process might seem unnecessary, but it is the hypothesis central to this project that this process
        will allow the model to retain memory of its environment and its own actions while simultaneously allowing it to 
        both act in a predictable, deterministic manner AND to allow it to modify or change its own behavior when presented
        with the same circumstances again.

        This means that, hopefully, this process will allow the model to have memory and to learn. For a full explanation of 
        why this might work, I'll be writing a white paper on this project soon.

        '''
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
        
        random_gen = np.random.default_rng()
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
        '''
        Create the sensation neurons.

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        The sensation neurons allow the user to give feedback to the model on how it is performing at its task 
        and on what its environment is like. It's up to the user to find create and useful implementations 
        for the sensation feature. Ideally, you would use it as a reward and punishment system, use it to indicate 
        the presence or absense of goals or threats, to allow the model to sense objects in its environment, etc. 
        This needs to be handled programmatically in the higher level codespace.
        '''
        self.sensations = []
        for i in range(0, self.num_sensations):
            self.sensations.append((np.random.randint(low=1, high=self.size)))
        return

    def __pos_sensation(self, sense_num: int, amt: int):
        '''
        Creates the positive sensation neurons

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        For use in the __new_sensations() function.
        '''

        current = 0
        tp.get(self.neurons, self.sensations[sense_num], current)
        tp.set(self.neurons, self.sensations[sense_num], amt + current)
        return
    
    def __neg_sensation(self, sense_num: int, amt: int):
        '''
        Creates the negative sensation neurons

        :Parameters:
        none

        :Returns:
        none

        :Comments:
        For use in the __new_sensations() function.
        '''
        current = 0
        tp.get(self.neurons, self.sensations[sense_num], current)
        tp.set(self.neurons, self.sensations[sense_num], amt + current)
        return
    
    def feedback(self, sense_num: int, amt: int, pos: bool):
        '''
        Give the model feedback on how it is performing at its task.

        :Parameters:
        sense_num (int): which sensation to use for this feedback.
        amt (int): how big or small of a sensation it is.
        pos(bool): whether this is a positive or negative feedback.

        :Returns:
        none

        :Comments:
        '''
        if (pos):
            self.__pos_sensation(sense_num, amt)
        else:
            self.__neg_sensation(sense_num, amt)
        return
    
    
    def permute(self, fraction: int):
        '''
        Permute the model's personality by a certain degree.

        :Parameters:
        fraction (int): positive integer which lessens the degree of the permutation as it receives higher values

        :Returns:
        none

        :Comments: 
        You will absolutely need to trial and error with the degree to see what works best for your use case.
        This function will enable iterating on the personality traits of a model which has already proven useful.
        You'll want to use this to make small, incremental improvements to a model and then test it to see whether to move 
        forward with the changes or roll back to a previous version.
        
        If you want the model to change quickly, set the fraction to 1. That will be the fastest it can change currently.
        If you want the model to change slowly (and in most cases you will want this), set the fraction
        to higher numbers. The higher fraction goes, the slower the model will change with each iteration.

        Once a minimal working model has been found, this function will be what we primarily use to iterate on it.
        '''
        model = general_dev()
        model.copy(self)
        model.__new_thresholds()
        model.__new_propensity()
        model.__new_personality()
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
        '''
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
        '''
        return
    
    def __convert_nparray(self, nparray: np.ndarray) -> tp.tensor:
        '''
        Converts a NumPy array to a TensorPlus tensor.

        :Parameters:
        nparray (numpy.ndarray): the array to be converted.

        :Returns:
        tensor (tensorplus.tensor): the converted tensor.

        :Comments:
        This is a bit of a slow process. Converting tensors to numpy arrays and vice versa is a significant
        performance bottleneck for the library currently. Eventually I'll want to replace everything 
        we're using NumPy for with TensorPlus functions, but we're not there yet.
        '''
        dims = nparray.shape
        size = dims[0] * dims[1]
        tensor = tp.create(size)
        for i in range(0, dims[0]):
            for j in range(0, dims[1]):
                tp.set(tensor, (i * dims[1]) + j, nparray[i, j])
        return tensor
    
    def update(self, input_image: np.ndarray) -> list[bool]:
        
        '''
        The main logic loop for using the model requires running this function for it to act.

        :Parameters:
        input_image: a NumPy array which is the same size as the model's image size

        :Returns:
        take_action: a list of booleans which represents which actions the model is taking with this update.
        
        :Comments:
        The logic for how this function works is as follows:
    
        Step 1
        Take the input image and resize it, then use it as starting values for the first layer
    
        Step 2
        Compare the neurons values to the positive and negative thresholds to determine which neurons are firing.

        Step 3
        Send the signals of each neuron to each of its connected neurons.

        Step 4
        Update the emotion layers by adding in their relevant personality layer. 

        Step 5 
        Check the control neurons to determine if any of them are firing. If they are, set it to true in the 
        output of the update function, if not, set it to false
        '''

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
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections2, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections3, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections4, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections5, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections6, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections7, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        tp.vector_add(self.neurons, self.__helper1, self.connections8, self.__helper2)
        tp.copy_t(self.__helper2, self.neurons)
        
        # now we repeat that process for the negative neurons
        tp.lesser(self.neurons, self.thresholds_neg, self.__helper1)
        
        # now update the negative thresholds, signals, and emotions based on which neurons fired
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
