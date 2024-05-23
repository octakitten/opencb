import torch
import numpy as np
import os
import sys

class general_dev():
    
    # records for how fast the model completes the game find_food_02
    min_dx = -1
    min_dy = -1

    # dimensions of the neural space
    width = 0
    height = 0
    depth = 0
    bounds = 0
    
    colors = 255
    
    maleability = 1.1

    range_high = 2.0
    range_low = 1/range_high
    
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
    '''personality9 = 0
    personality10 = 0
    personality11 = 0
    personality12 = 0
    personality13 = 0
    personality14 = 0
    personality15 = 0
    personality16 = 0
    '''
    
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

    propensity = 0
    

    def __init__(self):
        return
    
    def save(self, path):
        if (os.path.exists(path) == False):
            os.makedirs(path)
            try:
                os.makefile(path + '/width')
                os.makefile(path + '/height')
                os.makefile(path + '/depth')
                os.makefile(path + '/bounds')
                os.makefile(path + '/range_high')
                os.makefile(path + '/range_low')
                os.makefile(path + '/num_controls')
                os.makefile(path + '/controls')
                os.makefile(path + '/thresholds_pos.pth')
                os.makefile(path + '/thresholds_neg.pth')
                os.makefile(path + '/layer0.pth')
                os.makefile(path + '/layer1.pth')
                os.makefile(path + '/layer2.pth')
                os.makefile(path + '/layer3.pth')
                os.makefile(path + '/layer4.pth')
                os.makefile(path + '/layer5.pth')
                os.makefile(path + '/layer6.pth')
                os.makefile(path + '/layer7.pth')
                os.makefile(path + '/layer8.pth')
                os.makefile(path + '/layer9.pth')
                os.makefile(path + '/layer10.pth')
                os.makefile(path + '/layer11.pth')
                os.makefile(path + '/layer12.pth')
                os.makefile(path + '/layer13.pth')
                os.makefile(path + '/layer14.pth')
                os.makefile(path + '/layer15.pth')
                os.makefile(path + '/layer16.pth')
                os.makefile(path + '/layer17.pth')
                os.makefile(path + '/layer18.pth')
                os.makefile(path + '/layer19.pth')
                os.makefile(path + '/layer20.pth')
                os.makefile(path + '/layer21.pth')
                os.makefile(path + '/layer22.pth')
                os.makefile(path + '/layer23.pth')
                os.makefile(path + '/layer24.pth')
                os.makefile(path + '/layer25.pth')
                os.makefile(path + '/layer26.pth')
                os.makefile(path + '/layer27.pth')
                os.makefile(path + '/layer28.pth')
            except:
                print('error creating files')
        np.save(path + '/width', self.width)
        np.save(path + '/height', self.height)
        np.save(path + '/depth', self.depth)
        np.save(path + '/bounds', self.bounds)
        np.save(path + '/range_high', self.range_high)
        np.save(path + '/range_low', self.range_low)
        np.save(path + '/num_controls', self.num_controls) 
        np.save(path + '/controls.pth', self.controls)
        torch.save(self.thresholds_pos, path + '/thresholds_pos.pth')
        torch.save(self.thresholds_neg, path + 'thresholds_neg.pth')
        torch.save(self.layer0, path + '/layer0.pth')
        torch.save(self.layer1, path + '/layer1.pth')
        torch.save(self.layer2, path + '/layer2.pth')
        torch.save(self.layer3, path + '/layer3.pth')
        torch.save(self.layer4, path + '/layer4.pth')
        torch.save(self.emotion1, path + '/layer5.pth')
        torch.save(self.emotion2, path + '/layer6.pth')
        torch.save(self.emotion3, path + '/layer7.pth')
        torch.save(self.emotion4, path + '/layer8.pth')
        torch.save(self.emotion5, path + '/layer9.pth')
        torch.save(self.emotion6, path + '/layer10.pth')
        torch.save(self.emotion7, path + '/layer11.pth')
        torch.save(self.emotion8, path + '/layer12.pth')
        torch.save(self.personality1, path + '/layer13.pth')
        torch.save(self.personality2, path + '/layer14.pth')
        torch.save(self.personality3, path + '/layer15.pth')
        torch.save(self.personality4, path + '/layer16.pth')
        torch.save(self.personality5, path + '/layer17.pth')
        torch.save(self.personality6, path + '/layer18.pth')
        torch.save(self.personality7, path + '/layer19.pth')
        torch.save(self.personality8, path + '/layer20.pth')
        torch.save(self.personality9, path + '/layer21.pth')
        torch.save(self.personality10, path + '/layer22.pth')
        torch.save(self.personality11, path + '/layer23.pth')
        torch.save(self.personality12, path + '/layer24.pth')
        torch.save(self.personality13, path + '/layer25.pth')
        torch.save(self.personality14, path + '/layer26.pth')
        torch.save(self.personality15, path + '/layer27.pth')
        torch.save(self.personality16, path + '/layer28.pth')
        return
    
    def load(self, path):
        self.width = np.load(path + '/width.pth')
        self.height = np.load(path + '/height.pth')
        self.depth = np.load(path + '/depth.pth')
        self.bounds = np.load(path + '/bounds.pth')
        self.range_high = np.load(path + '/range_high.pth')
        self.range_low = np.load(path + '/range_low.pth')
        self.num_controls = np.load(path + '/num_controls.pth')
        self.controls = np.load(path + '/controls.pth')
        self.thresholds_pos = torch.load(path + '/thresholds_pos.pth')
        self.thresholds_neg = torch.load(path + '/thresholds_neg.pth')
        self.layer0 = torch.load(path + '/layer0.pth')
        self.layer1 = torch.load(path + '/layer1.pth')
        self.layer2 = torch.load(path + '/layer2.pth')
        self.layer3 = torch.load(path + '/layer3.pth')
        self.layer4 = torch.load(path + '/layer4.pth')
        self.emotion1 = torch.load(path + '/layer5.pth')
        self.emotion2 = torch.load(path + '/layer6.pth')
        self.emotion3 = torch.load(path + '/layer7.pth')
        self.emotion4 = torch.load(path + '/layer8.pth')
        self.emotion5 = torch.load(path + '/layer9.pth')
        self.emotion6 = torch.load(path + '/layer10.pth')
        self.emotion7 = torch.load(path + '/layer11.pth')
        self.emotion8 = torch.load(path + '/layer12.pth')
        self.personality1 = torch.load(path + '/layer13.pth')
        self.personality2 = torch.load(path + '/layer14.pth')
        self.personality3 = torch.load(path + '/layer15.pth')
        self.personality4 = torch.load(path + '/layer16.pth')
        self.personality5 = torch.load(path + '/layer17.pth')
        self.personality6 = torch.load(path + '/layer18.pth')
        self.personality7 = torch.load(path + '/layer19.pth')
        self.personality8 = torch.load(path + '/layer20.pth')
        self.personality9 = torch.load(path + '/layer21.pth')
        self.personality10 = torch.load(path + '/layer22.pth')
        self.personality11 = torch.load(path + '/layer23.pth')
        self.personality12 = torch.load(path + '/layer24.pth')
        self.personality13 = torch.load(path + '/layer25.pth')
        self.personality14 = torch.load(path + '/layer26.pth')
        self.personality15 = torch.load(path + '/layer27.pth')
        self.personality16 = torch.load(path + '/layer28.pth')
        
    def copy(self, model):
        '''
        Copy a model's parameters to a new model.
        
        :Parameters:
        model (general): the model to copy
        
        :Returns:
        none
        '''
        self.width = model.width
        self.height = model.height
        self.depth = model.depth
        self.bounds = model.bounds
        self.range_high = model.range_high
        self.range_low = model.range_low
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
    
    def create(self, w, h, d, b, num_controls, num_sensations):    
        '''
        Create a new model with the given dimensions and number of controls.
        
        :Parameters: 
        w (int): width of input images in pixels
        h (int): height of input images in pixels
        d (int): depth of the neural space
        num_controls (int): number of controls
        
        :Returns:
        none
        
        :Comments:
        This function creates a new, randomly initialized model with the dimensions and number of controls given.
        Importantly, this model will only be able to accept images of the specified width and height.
        The depth of the model determines its complexity. With more depth, the runtime and memory usage
        also increase dramatically. The number of controls determines what outputs the model can have. If you want it to 
        perform a certain task that requires, for instance, controlling 4 seperate keyboard keypresses, 
        then you would want a model with 4 controls.
        '''
        self.width = w
        print('assigned width')
        self.height = h
        print('assigned height')
        self.depth = d
        print('assigned depth')
        self.bounds = b
        print('assigned bounds')
        self.num_controls = num_controls
        print('assigned controls')
        self.num_sensations = num_sensations
        print('assigned values')
        self.__new_controls()
        print('new controls')
        self.__new_thresholds()
        print('new thresholds')
        self.__new_propensity()
        print('new propensity')
        self.__new_personality()
        print('new personality')
        self.__new_sensations()
        print('new sensations')
        return
   
    def __new_range(self, r): 
        self.range_low = np.arctan(2*r/np.pi) + 2 
        self.range_high = 1 / self.range_low
        if (self.range_low > self.range_high):
            self.range_low, self.range_high = self.range_high, self.range_low
        return
    
    def __new_thresholds(self):
        # threshhold_max = np.random.uniform(low=self.range_low, high=self.range_high)
        random_gen = torch.Generator(device=torch.device('cuda'))
        random_gen.seed()
        self.thresholds_pos = torch.tensor(data=1, device=torch.device('cuda'))
        self.thresholds_neg = torch.tensor(data=1, device=torch.device('cuda'))
        self.thresholds_pos = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=torch.device('cuda'))
        torch.add(torch.mul(self.thresholds_pos, self.bounds, out=self.thresholds_pos), 1, out=self.thresholds_pos)
        random_gen.seed()
        self.thresholds_neg = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=torch.device('cuda'))
        torch.subtract(1, torch.divide(self.thresholds_neg, self.bounds, out=self.thresholds_neg), out=self.thresholds_neg)

        #self.thresholds_pos = torch.rand()
        #for i in range(0, self.num_controls):
        #    self.thresholds_pos.append(np.random.uniform(low=1, high=threshhold_max))
        #    self.thresholds_neg.append(np.random.uniform(low=self.range_low, high=1))
        return
    
    def __new_controls(self):
        self.controls = []
        for i in range(0, self.num_controls):
            self.controls.append((np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), np.random.randint(low=1, high=self.depth)))
        return
    
    def __new_personality(self):
            
        '''
        Initialize the model's personality layers.

        :Parameters:
        none

        :Returns:
        none

        :Comments: 
        the personality layers are the only parts of the model that don't change over time. we initialize all the layers here,
        from layer0 to emotion8 to personality8, but the personality layers we initialize to random values. These random values
        should range from 1 to n for the positive personality layers, and 1 to 1/n for the negative personality layers. in order to 
        achieve this, we first generate random values between 0 and 1, then for the positive layers we multiply by n and add 1, and for
        the negative layers we divide by n and subtract from 1. This will give us the desired range of values for the personality layers.
        '''
        self.layer0 = torch.tensor(data=1, device=torch.device('cuda'))
        self.layer1 = torch.tensor(data=1, device=torch.device('cuda'))
        self.layer2 = torch.tensor(data=1, device=torch.device('cuda'))
        self.layer3 = torch.tensor(data=1, device=torch.device('cuda'))
        self.layer4 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion1 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion2 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion3 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion4 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion5 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion6 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion7 = torch.tensor(data=1, device=torch.device('cuda'))
        self.emotion8 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality1 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality2 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality3 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality4 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality5 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality6 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality7 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality8 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality9 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality10 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality11 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality12 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality13 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality14 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality15 = torch.tensor(data=1, device=torch.device('cuda'))
        self.personality16 = torch.tensor(data=1, device=torch.device('cuda'))
        '''
        print(self.layer0)
        print(self.emotion1)
        print(self.personality1)
        '''
        # neuron layer
        self.layer0 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        # threshold layers
        self.layer1 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.layer2 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        #multiplier layers
        self.layer3 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.layer4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        # emotion layers
        self.emotion1 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion2 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion3 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion5 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion6 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion7 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.emotion8 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        
        # personality layers
        self.personality1 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality2 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality3 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality5 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality6 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality7 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality8 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality9 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality10 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality11 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality12 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality13 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality14 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality15 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.personality16 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16,device=torch.device('cuda'))
        
        #print(self.layer0)
        #print(self.emotion1)
        #print(self.personality1)
        self.positive_firing= torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.positive_resting = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.negative_firing = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.negative_resting = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.pos_fire_amt = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.neg_fire_amt = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.pos_fire_amt_mult = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        self.neg_fire_amt_mult = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=torch.device('cuda'))
        
        # personality layers
        # positive thresh firing is used
        random_gen = torch.Generator(device=torch.device('cuda'))
        random_gen.seed()
        self.personality1 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality2 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality3 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality4 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality5 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality6 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality7 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality8 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality9 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality10 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality11 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality12 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality13 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality14 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality15 = torch.multiply(other=self.pos_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        self.personality16 = torch.multiply(other=self.neg_propensity[0,0], input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=torch.device('cuda'))).to(dtype=torch.int16)
        random_gen.seed()
        #print(self.layer0)
        #print(self.emotion1)
        #print(self.personality1)
        return
    
    def __new_propensity(self):
        random_gen = torch.Generator(device=torch.device('cuda'))
        random_gen.seed()
        self.pos_propensity = torch.tensor(data=1, device=torch.device('cuda'))
        self.neg_propensity = torch.tensor(data=1, device=torch.device('cuda'))
        self.pos_propensity = torch.rand(size=(2,2), generator=random_gen, device=torch.device('cuda'))
        self.pos_propensity = torch.add(torch.mul(self.pos_propensity, self.bounds, out=self.pos_propensity), 1).to(dtype=torch.int16)
        random_gen.seed()
        self.neg_propensity = torch.rand(size=(2, 2), generator=random_gen, device=torch.device('cuda'))
        self.neg_propensity = torch.subtract(-1, torch.mul(self.neg_propensity, self.bounds, out=self.neg_propensity)).to(dtype=torch.int16)
        #nep = torch.rand(size=(2,2), generator=random_gen, dtype=torch.int16, device=torch.device('cuda'))
        #pep = torch.ones(size=(2,2), device=torch.device('cuda'))
        #self.pos_propensity = torch.divide(nep[1,:], pep[1,:])
        #self.neg_propensity = nep[0,:]
        #nep = np.random.uniform(low=self.range_low, high=1)
        #pep = np.divide(1, self.neg_propensity)
        #prop = torch.Tensor([nep, pep])
        #self.propensity = torch.Tensor()
        #self.neg_propensity = nep
        #self.pos_propensity = pep
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
    
    def train(self, sense_num, amt, pos):    
        '''
        Train the model by giving it feedback on its actions.

        :Parameters:
        sense_num (int): the index of the sensation neuron to train
        amt (float): the amount to train the sensation neuron by
        pos (bool): whether the sensation is positive or negative

        :Returns:
        none

        :Comments: 
        Call this function whenever the model either does something right or makes a mistake.
        set pos to True if the sensation is positive, and False if the sensation is negative.
        You'll need to set conditions in your game that call this function automatically while it's playing.
        This function is also intended to be used later on in the training process, when the model is
        being used by a user on real world tasks and needs feedback.
        '''
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
        model = general_dev()
        model.copy(self)
        model.__new_thresholds()
        model.__new_propensity()
        model.__new_personality()
        torch.divide(model.thresholds_pos, fraction, out=model.thresholds_pos)
        torch.divide(model.thresholds_neg, fraction, out=model.thresholds_neg)
        model.personality1 = torch.divide(model.personality1, fraction,).to(dtype=torch.int16)
        model.personality2 = torch.divide(model.personality2, fraction,).to(dtype=torch.int16)
        model.personality3 = torch.divide(model.personality3, fraction,).to(dtype=torch.int16)
        model.personality4 = torch.divide(model.personality4, fraction,).to(dtype=torch.int16)
        model.personality5 = torch.divide(model.personality5, fraction,).to(dtype=torch.int16)
        model.personality6 = torch.divide(model.personality6, fraction,).to(dtype=torch.int16)
        model.personality7 = torch.divide(model.personality7, fraction,).to(dtype=torch.int16)
        model.personality8 = torch.divide(model.personality8, fraction,).to(dtype=torch.int16)
        model.personality9 = torch.divide(model.personality9, fraction,).to(dtype=torch.int16)
        model.personality10 = torch.divide(model.personality10, fraction,).to(dtype=torch.int16)
        model.personality11 = torch.divide(model.personality11, fraction,).to(dtype=torch.int16)
        model.personality12 = torch.divide(model.personality12, fraction,).to(dtype=torch.int16)
        model.personality13 = torch.divide(model.personality13, fraction,).to(dtype=torch.int16)
        model.personality14 = torch.divide(model.personality14, fraction,).to(dtype=torch.int16)
        model.personality15 = torch.divide(model.personality15, fraction,).to(dtype=torch.int16)
        model.personality16 = torch.divide(model.personality16, fraction,).to(dtype=torch.int16)
        '''
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
        torch.divide(model.personality16, fraction, out=model.personality16)'''
        for i in range(0, degree):
            torch.add(self.thresholds_pos, model.thresholds_pos, out=self.thresholds_pos)
            torch.add(self.thresholds_neg, model.thresholds_neg, out=self.thresholds_neg)
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
        deg = degree / fraction
        torch.divide(self.thresholds_pos, deg + 1, out=self.thresholds_pos)
        torch.divide(self.thresholds_neg, deg + 1, out=self.thresholds_neg)
        self.personality1 = torch.divide(self.personality1, deg + 1).to(dtype=torch.int16)
        self.personality2 = torch.divide(self.personality2, deg + 1).to(dtype=torch.int16)
        self.personality3 = torch.divide(self.personality3, deg + 1).to(dtype=torch.int16)
        self.personality4 = torch.divide(self.personality4, deg + 1).to(dtype=torch.int16)
        self.personality5 = torch.divide(self.personality5, deg + 1).to(dtype=torch.int16)
        self.personality6 = torch.divide(self.personality6, deg + 1).to(dtype=torch.int16)
        self.personality7 = torch.divide(self.personality7, deg + 1).to(dtype=torch.int16)
        self.personality8 = torch.divide(self.personality8, deg + 1).to(dtype=torch.int16)
        self.personality9 = torch.divide(self.personality9, deg + 1).to(dtype=torch.int16)
        self.personality10 = torch.divide(self.personality10, deg + 1).to(dtype=torch.int16)
        self.personality11 = torch.divide(self.personality11, deg + 1).to(dtype=torch.int16)
        self.personality12 = torch.divide(self.personality12, deg + 1).to(dtype=torch.int16)
        self.personality13 = torch.divide(self.personality13, deg + 1).to(dtype=torch.int16)
        self.personality14 = torch.divide(self.personality14, deg + 1).to(dtype=torch.int16)
        self.personality15 = torch.divide(self.personality15, deg + 1).to(dtype=torch.int16)
        self.personality16 = torch.divide(self.personality16, deg + 1).to(dtype=torch.int16)
        '''
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
        '''
        return

    # @TODO: NaNs... NaNs everywhere!
    def update(self, input_image):
        '''
        Main control function for the model.

        :Parameters:
        input_image (tensor): the image to input into the model

        :Returns:
        take_action (list): a list of booleans representing whether the controls should be activated or not

        :Comments: 
        This function is what makes the model 'act'. It takes an image as input and processes it by firing neurons.
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

        if (torch.is_tensor(input_image) == False):
            return -1
        # add in the input image
        input_image.to(dtype=torch.int16, device=torch.device('cuda'))
        input_tensor = torch.tensor(data=1, device=torch.device('cuda'))
        
        input_tensor = torch.clone(input_image, ).detach()
        #print(input_image.device)
        #print(input_tensor)
        #torch.div(input_tensor, 255, out=input_tensor)
        #torch.mul(input_tensor, self.bounds, out=input_tensor)
        #torch.add(input_tensor, torch.ones(size=input_image.size(), device=torch.device('cuda')), out=input_tensor)
        #print(input_tensor)
        #print('layer0')
        #print(self.layer0)
        torch.add(self.layer0[:, :, 1],  input_tensor, out=self.layer0[:, :, 1])
        #print("1")
        #print('layer0')
        #print(self.layer0)
        #print(self.layer0)

        #check which neurons are firing and which arent, do the stuff
        torch.greater(self.layer0, self.layer1, out=self.positive_firing)
        torch.less_equal(self.layer0, self.layer1, out=self.positive_resting)
        torch.greater(self.layer0, self.layer2, out=self.negative_firing)
        torch.less_equal(self.layer0, self.layer2, out=self.negative_resting)
        #print("2")
        #print('layer0')
        #print(self.layer0)

        # keep track of the threshold values of the firing neurons
        torch.multiply(self.positive_firing, self.layer1, out=self.pos_fire_amt)
        torch.multiply(self.negative_firing, self.layer2, out=self.neg_fire_amt)
        #print('2 - 1')
        #print(self.positive_firing)
        #print(self.negative_firing)
        #print(self.pos_fire_amt)
        #print(self.neg_fire_amt)
        
        self.pos_fre_amt = torch.div(self.pos_fire_amt, 6).to(dtype=torch.int16)
        self.neg_fire_amt = torch.div(self.neg_fire_amt, 6).to(dtype=torch.int16)
        
        # use the firing multipliers to change the output values of the firing neurons
        torch.add(self.pos_fire_amt, self.layer3, out=self.pos_fire_amt_mult)
        torch.sub(self.neg_fire_amt, self.layer4, out=self.neg_fire_amt_mult)
        #print('2 - 2')
        #print(self.pos_fire_amt_mult)
        #print(self.neg_fire_amt_mult)
        '''
        pos_firing_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=torch.device('cuda'), dtype=torch.int16)
        pos_resting_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=torch.device('cuda'), dtype=torch.int16)
        neg_firing_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=torch.device('cuda'), dtype=torch.int16)
        neg_resting_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=torch.device('cuda'), dtype=torch.int16)
        torch.logical_not(self.positive_firing, out=pos_firing_NOT)
        torch.logical_not(self.positive_resting, out=pos_resting_NOT)
        torch.logical_not(self.negative_firing, out=neg_firing_NOT)
        torch.logical_not(self.negative_resting, out=neg_resting_NOT)
        print('2 - 3')
        print(pos_firing_NOT)
        print(pos_resting_NOT)
        print(neg_firing_NOT)
        print(neg_resting_NOT)'''

        #ensure that default values are 1 and not 0 for multiplication and division purposes from here forward
        #torch.add(self.pos_fire_amt_mult, pos_firing_NOT, out=self.pos_fire_amt_mult)
        #torch.add(self.neg_fire_amt_mult, neg_firing_NOT, out=self.neg_fire_amt_mult)
        #print('2 - 4')
        #print(self.pos_fire_amt_mult)
        #print(self.neg_fire_amt_mult)
        #print('layer0')
        #print(self.layer0)

        # apply the firing values to each of the near neighbors
        temp = torch.zeros(size=(self.width, self.height, self.depth), device=torch.device('cuda'), dtype=torch.int16)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, 1, 0), out=temp)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, -1, 0), out=temp)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, 1, 1), out=temp)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, -1, 1), out=temp)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, 1, 2), out=temp)
        torch.add(self.layer0, torch.roll(self.pos_fire_amt_mult, -1, 2), out=temp)
        torch.sub(self.layer0, torch.roll(self.neg_fire_amt_mult, 1, 0), out=temp)
        torch.sub(self.layer0, torch.roll(self.neg_fire_amt_mult, -1, 0), out=temp)
        torch.sub(self.layer0, torch.roll(self.neg_fire_amt_mult, 1, 1), out=temp)
        torch.sub(self.layer0, torch.roll(self.neg_fire_amt_mult, -1, 1), out=temp)
        torch.sub(self.layer0, torch.roll(self.neg_fire_amt_mult, 1, 2), out=temp)
        torch.sub(self.layer0, torch.roll(self.neg_fire_amt_mult, -1, 2), out=temp)
        torch.add(self.layer0, temp, out=self.layer0)
        '''print("3")
        print('layer0')
        print(self.layer0)'''
        
        # check the predefined output neurons to see if they're ready to fire
        # if they are, then return the action(s) to take
        take_action = []
        
        #print(self.layer0)
        #print('layer0')
        for i in range(0, self.num_controls):
            #print(self.layer0[self.controls[i]])
            if (self.layer0[self.controls[i]].item() > self.thresholds_pos[i, 0].item()):
                take_action.append(True)
                self.layer0[self.controls[i]] = self.layer0[self.controls[i]] - self.thresholds_pos[i,0]
            else:
                if (self.layer0[self.controls[i]].item() > self.thresholds_neg[i,0].item()):
                    take_action.append(False)
                else:
                    take_action.append(-1)
        
        # update layer0 by setting all the firing neurons to 0
        torch.mul(self.layer0, torch.logical_not(self.positive_firing), out=self.layer0)
        torch.mul(self.layer0, torch.logical_not(self.negative_firing), out=self.layer0)

        # update the threshold layers
        torch.add(torch.mul(self.positive_firing, self.emotion1), self.layer1, out=self.layer1)
        torch.add(torch.mul(self.positive_resting, self.emotion2), self.layer1, out=self.layer1)
        torch.add(torch.mul(self.negative_firing, self.emotion3), self.layer2, out=self.layer2)
        torch.add(torch.mul(self.negative_resting, self.emotion4), self.layer2, out=self.layer2)
        torch.add(torch.mul(self.positive_firing, self.emotion5), self.layer3, out=self.layer3)
        torch.add(torch.mul(self.positive_resting, self.emotion6), self.layer3, out=self.layer3)
        torch.add(torch.mul(self.negative_firing, self.emotion7), self.layer4, out=self.layer4)
        torch.add(torch.mul(self.negative_resting, self.emotion8), self.layer4, out=self.layer4)
        
        '''
        #temp = torch.zeros(size=(self.width, self.height, self.depth), device=torch.device('cuda'), dtype=torch.int16)
        
        #torch.mul(self.layer1, torch.mul(self.emotion1, torch.add(self.positive_firing, pos_firing_NOT)), out=self.layer1)
        #torch.mul(self.layer1, torch.mul(self.emotion2, torch.add(self.positive_resting, pos_resting_NOT)), out=self.layer1)
        #torch.mul(self.layer2, torch.mul(self.emotion3, torch.add(self.negative_firing, neg_firing_NOT)), out=self.layer2)
        #torch.mul(self.layer2, torch.mul(self.emotion4, torch.add(self.negative_resting, neg_firing_NOT)), out=self.layer2)
        #torch.mul(self.layer3, torch.mul(self.emotion5, torch.add(self.positive_firing, pos_firing_NOT)), out=self.layer3)
        #torch.mul(self.layer3, torch.mul(self.emotion6, torch.add(self.positive_resting, pos_resting_NOT)), out=self.layer3)
        #torch.mul(self.layer4, torch.mul(self.emotion7, torch.add(self.negative_firing, neg_firing_NOT)), out=self.layer4)
        #torch.mul(self.layer4, torch.mul(self.emotion8, torch.add(self.negative_resting, neg_firing_NOT)), out=self.layer4)
        
        #torch.add(torch.multiply(self.positive_firing, self.emotion1), pos_firing_NOT, out=temp)
        torch.add(torch.multiply(self.positive_resting, self.emotion2), pos_resting_NOT, out=temp)
        torch.mul(self.layer1, temp, out=self.layer1)
        torch.add(torch.multiply(self.negative_firing, self.emotion3), neg_firing_NOT, out=temp)
        torch.mul(self.layer2, temp, out=self.layer2)
        torch.add(torch.multiply(self.negative_resting, self.emotion4), neg_resting_NOT, out=temp)
        torch.mul(self.layer2, temp, out=self.layer2)
        
        # update the emotion (multiplier) layers
        torch.add(torch.multiply(self.positive_firing, self.emotion5), pos_firing_NOT, out=temp)
        torch.mul(self.layer3, temp, out=self.layer3)
        torch.add(torch.multiply(self.positive_resting, self.emotion6), pos_resting_NOT, out=temp)
        torch.mul(self.layer3, temp, out=self.layer3)
        torch.add(torch.multiply(self.negative_firing, self.emotion7), neg_firing_NOT, out=temp)
        torch.mul(self.layer4, temp, out=self.layer4)
        torch.add(torch.multiply(self.negative_resting, self.emotion8), neg_resting_NOT, out=temp)
        torch.mul(self.layer4, temp, out=self.layer4)
        '''
        # figure out which emotions were used and which weren't
        # and then update them according to the personality values
        
        torch.add(torch.mul(self.positive_firing, self.personality1), self.emotion1, out=self.emotion1)
        torch.add(torch.mul(self.positive_resting, self.personality2), self.emotion1, out=self.emotion1)
        torch.add(torch.mul(self.negative_firing, self.personality3), self.emotion2, out=self.emotion2)
        torch.add(torch.mul(self.negative_resting, self.personality4), self.emotion2, out=self.emotion2)
        torch.add(torch.mul(self.positive_firing, self.personality5), self.emotion3, out=self.emotion3)
        torch.add(torch.mul(self.positive_resting, self.personality6), self.emotion3, out=self.emotion3)
        torch.add(torch.mul(self.negative_firing, self.personality7), self.emotion4, out=self.emotion4)
        torch.add(torch.mul(self.negative_resting, self.personality8), self.emotion4, out=self.emotion4)
        torch.add(torch.mul(self.positive_firing, self.personality9), self.emotion5, out=self.emotion5)
        torch.add(torch.mul(self.positive_resting, self.personality10), self.emotion5, out=self.emotion5)
        torch.add(torch.mul(self.negative_firing, self.personality11), self.emotion6, out=self.emotion6)
        torch.add(torch.mul(self.negative_resting, self.personality12), self.emotion6, out=self.emotion6)
        torch.add(torch.mul(self.positive_firing, self.personality13), self.emotion7, out=self.emotion7)
        torch.add(torch.mul(self.positive_resting, self.personality14), self.emotion7, out=self.emotion7)
        torch.add(torch.mul(self.negative_firing, self.personality15), self.emotion8, out=self.emotion8)
        torch.add(torch.mul(self.negative_resting, self.personality16), self.emotion8, out=self.emotion8)
        
        # then do the same for the mult layers
        
        
        '''
        torch.mul(self.emotion1, torch.add(torch.multiply(self.positive_firing, self.personality1), pos_firing_NOT), out=self.emotion1)
        torch.mul(self.emotion2, torch.add(torch.multiply(self.positive_resting, self.personality2), pos_resting_NOT), out=self.emotion2)
        torch.mul(self.emotion3, torch.add(torch.multiply(self.negative_firing, self.personality3), neg_firing_NOT), out=self.emotion3)
        torch.mul(self.emotion4, torch.add(torch.multiply(self.negative_resting, self.personality4), neg_resting_NOT), out=self.emotion4)
        torch.mul(self.emotion5, torch.add(torch.multiply(self.positive_firing, self.personality5), pos_firing_NOT), out=self.emotion5)
        torch.mul(self.emotion6, torch.add(torch.multiply(self.positive_resting, self.personality6), pos_resting_NOT), out=self.emotion6)
        torch.mul(self.emotion7, torch.add(torch.multiply(self.negative_firing, self.personality7), neg_firing_NOT), out=self.emotion7)
        torch.mul(self.emotion8, torch.add(torch.multiply(self.negative_resting, self.personality8), neg_resting_NOT), out=self.emotion8)
        #torch.mul(torch.multiply(self.positive_resting, self.personality2), torch.multiply(self.positive_firing, self.personality4), out=self.emotion2)
        #torch.mul(torch.multiply(self.negative_firing, self.personality5), torch.multiply(self.negative_resting, self.personality7), out=self.emotion3)
        #torch.mul(torch.multiply(self.negative_resting, self.personality6), torch.multiply(self.negative_firing, self.personality8), out=self.emotion4)
        
        # do the same for the emotion multipliers
        #torch.mul(torch.multiply(self.positive_firing, self.personality9), torch.multiply(self.positive_resting, self.personality11), out=self.emotion5)
        #torch.mul(torch.multiply(self.positive_resting, self.personality10), torch.multiply(self.positive_firing, self.personality12), out=self.emotion6)
        #torch.mul(torch.multiply(self.negative_firing, self.personality13), torch.multiply(self.negative_resting, self.personality15), out=self.emotion7)
        #torch.mul(torch.multiply(self.negative_resting, self.personality14), torch.multiply(self.negative_firing, self.personality16), out=self.emotion8)
        torch.mul(self.positive_firing, self.maleability, out=self.positive_firing)
        torch.div(self.positive_resting, self.maleability, out=self.positive_resting)
        torch.mul(self.negative_firing, self.maleability, out=self.negative_firing)
        torch.div(self.negative_resting, self.maleability, out=self.negative_resting)
        torch.add(self.positive_firing, pos_firing_NOT, out=self.positive_firing)
        torch.add(self.positive_resting, pos_resting_NOT, out=self.positive_resting)
        torch.add(self.negative_firing, neg_firing_NOT, out=self.negative_firing)
        torch.add(self.negative_resting, neg_resting_NOT, out=self.negative_resting)
        torch.mul(self.positive_firing, self.personality1, out=self.personality1)
        torch.mul(self.positive_resting, self.personality2, out=self.personality2)
        torch.mul(self.negative_firing, self.personality3, out=self.personality3)
        torch.mul(self.negative_resting, self.personality4, out=self.personality4)
        torch.mul(self.positive_firing, self.personality5, out=self.personality5)
        torch.mul(self.positive_resting, self.personality6, out=self.personality6)
        torch.mul(self.negative_firing, self.personality7, out=self.personality7)
        torch.mul(self.negative_resting, self.personality8, out=self.personality8)
        
        # once a neuron has fired, its value needs to be lowered by the threshold amount
        torch.add(self.pos_fire_amt, pos_firing_NOT, out=self.pos_fire_amt)
        torch.add(self.neg_fire_amt, neg_firing_NOT, out=self.neg_fire_amt)
        torch.divide(self.layer0, self.pos_fire_amt, out=self.layer0)
        torch.mul(self.layer0, self.neg_fire_amt, out=self.layer0)
        input()'''
        #print(take_action)
        return take_action