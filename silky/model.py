import torch
import numpy as np
import os

class general_dev2():
        
    device = -1
    
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
    personality9 = 0
    personality10 = 0
    personality11 = 0
    personality12 = 0
    personality13 = 0
    personality14 = 0
    personality15 = 0
    personality16 = 0
    
    # dna layers
    dna1 = 0
    dna2 = 0
    dna3 = 0
    dna4 = 0
    dna5 = 0
    dna6 = 0
    dna7 = 0
    dna8 = 0
    dna9 = 0
    dna10 = 0
    dna11 = 0
    dna12 = 0
    dna13 = 0
    dna14 = 0
    dna15 = 0
    dna16 = 0
    dna17 = 0
    dna18 = 0
    dna19 = 0
    dna20 = 0
    dna21 = 0
    dna22 = 0
    dna23 = 0
    dna24 = 0
    dna25 = 0
    dna26 = 0
    dna27 = 0
    dna28 = 0
    dna29 = 0
    dna30 = 0
    dna31 = 0
    dna32 = 0
    
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
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return

    def __check_cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return
    
    def save(self, path):
        if (os.path.exists(path) == False):
            os.makedirs(path)
            try:
                '''
                os.makefile(path + '/width')
                os.makefile(path + '/height')
                os.makefile(path + '/depth')
                os.makefile(path + '/bounds')
                os.makefile(path + '/range_high')
                os.makefile(path + '/range_low')
                os.makefile(path + '/num_controls')
                os.makefile(path + '/controls')
                os.makefile(path + '/num_sensations')
                os.makefile(path + '/sensations')
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
                '''
            except:
                print('error creating files')
        np.save(path + '/width', self.width)
        np.save(path + '/height', self.height)
        np.save(path + '/depth', self.depth)
        np.save(path + '/bounds', self.bounds)
        np.save(path + '/range_high', self.range_high)
        np.save(path + '/range_low', self.range_low)
        np.save(path + '/num_controls', self.num_controls) 
        np.save(path + '/controls', self.controls)
        np.save(path + '/num_sensations', self.num_sensations)
        np.save(path + '/sensations', self.sensations)
        torch.save(self.thresholds_pos, path + '/thresholds_pos.pth')
        torch.save(self.thresholds_neg, path + '/thresholds_neg.pth')
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
        torch.save(self.dna1, path + '/layer29.pth')
        torch.save(self.dna2, path + '/layer30.pth')
        torch.save(self.dna3, path + '/layer31.pth')
        torch.save(self.dna4, path + '/layer32.pth')
        torch.save(self.dna5, path + '/layer33.pth')
        torch.save(self.dna6, path + '/layer34.pth')
        torch.save(self.dna7, path + '/layer35.pth')
        torch.save(self.dna8, path + '/layer36.pth')
        torch.save(self.dna9, path + '/layer37.pth')
        torch.save(self.dna10, path + '/layer38.pth')
        torch.save(self.dna11, path + '/layer39.pth')
        torch.save(self.dna12, path + '/layer40.pth')
        torch.save(self.dna13, path + '/layer41.pth')
        torch.save(self.dna14, path + '/layer42.pth')
        torch.save(self.dna15, path + '/layer43.pth')
        torch.save(self.dna16, path + '/layer44.pth')
        torch.save(self.dna17, path + '/layer45.pth')
        torch.save(self.dna18, path + '/layer46.pth')
        torch.save(self.dna19, path + '/layer47.pth')
        torch.save(self.dna20, path + '/layer48.pth')
        torch.save(self.dna21, path + '/layer49.pth')
        torch.save(self.dna22, path + '/layer50.pth')
        torch.save(self.dna23, path + '/layer51.pth')
        torch.save(self.dna24, path + '/layer52.pth')
        torch.save(self.dna25, path + '/layer53.pth')
        torch.save(self.dna26, path + '/layer54.pth')
        torch.save(self.dna27, path + '/layer55.pth')
        torch.save(self.dna28, path + '/layer56.pth')
        torch.save(self.dna29, path + '/layer57.pth')
        torch.save(self.dna30, path + '/layer58.pth')
        torch.save(self.dna31, path + '/layer59.pth')
        torch.save(self.dna32, path + '/layer60.pth')
        return
    
    def load(self, path):
        self.__check_cuda()
        self.width = np.load(path + '/width.npy')
        print(self.width)
        self.height = np.load(path + '/height.npy')
        self.depth = np.load(path + '/depth.npy')
        self.bounds = np.load(path + '/bounds.npy')
        try:
            self.bounds = self.bounds.item()
        except:
            print('error converting bounds to int')
        self.range_high = np.load(path + '/range_high.npy')
        self.range_low = np.load(path + '/range_low.npy')
        self.num_controls = np.load(path + '/num_controls.npy')
        self.controls = np.load(path + '/controls.npy')
        self.num_sensations = np.load(path + '/num_sensations.npy')
        self.sensations = np.load(path + '/sensations.npy')
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
        self.dna1 = torch.load(path + '/layer29.pth')
        self.dna2 = torch.load(path + '/layer30.pth')
        self.dna3 = torch.load(path + '/layer31.pth')
        self.dna4 = torch.load(path + '/layer32.pth')
        self.dna5 = torch.load(path + '/layer33.pth')
        self.dna6 = torch.load(path + '/layer34.pth')
        self.dna7 = torch.load(path + '/layer35.pth')
        self.dna8 = torch.load(path + '/layer36.pth')
        self.dna9 = torch.load(path + '/layer37.pth')
        self.dna10 = torch.load(path + '/layer38.pth')
        self.dna11 = torch.load(path + '/layer39.pth')
        self.dna12 = torch.load(path + '/layer40.pth')
        self.dna13 = torch.load(path + '/layer41.pth')
        self.dna14 = torch.load(path + '/layer42.pth')
        self.dna15 = torch.load(path + '/layer43.pth')
        self.dna16 = torch.load(path + '/layer44.pth')
        self.dna17 = torch.load(path + '/layer45.pth')
        self.dna18 = torch.load(path + '/layer46.pth')
        self.dna19 = torch.load(path + '/layer47.pth')
        self.dna20 = torch.load(path + '/layer48.pth')
        self.dna21 = torch.load(path + '/layer49.pth')
        self.dna22 = torch.load(path + '/layer50.pth')
        self.dna23 = torch.load(path + '/layer51.pth')
        self.dna24 = torch.load(path + '/layer52.pth')
        self.dna25 = torch.load(path + '/layer53.pth')
        self.dna26 = torch.load(path + '/layer54.pth')
        self.dna27 = torch.load(path + '/layer55.pth')
        self.dna28 = torch.load(path + '/layer56.pth')
        self.dna29 = torch.load(path + '/layer57.pth')
        self.dna30 = torch.load(path + '/layer58.pth')
        self.dna31 = torch.load(path + '/layer59.pth')
        self.dna32 = torch.load(path + '/layer60.pth')

        self.positive_firing= torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.positive_resting = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.negative_firing = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.negative_resting = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.pos_fire_amt = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.neg_fire_amt = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.pos_fire_amt_mult = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.neg_fire_amt_mult = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        
        
    def copy(self, model):
        '''
        Copy a model's parameters to a new model.
        
        :Parameters:
        model (general_dev2): the model to copy
        
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
        self.layer3 = model.layer3
        self.layer4 = model.layer4
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
        self.pos_propensity = model.pos_propensity
        self.neg_propensity = model.neg_propensity
        self.dna1 = model.dna1
        self.dna2 = model.dna2
        self.dna3 = model.dna3
        self.dna4 = model.dna4
        self.dna5 = model.dna5
        self.dna6 = model.dna6
        self.dna7 = model.dna7
        self.dna8 = model.dna8
        self.dna9 = model.dna9
        self.dna10 = model.dna10
        self.dna11 = model.dna11
        self.dna12 = model.dna12
        self.dna13 = model.dna13
        self.dna14 = model.dna14
        self.dna15 = model.dna15
        self.dna16 = model.dna16
        self.dna17 = model.dna17
        self.dna18 = model.dna18
        self.dna19 = model.dna19
        self.dna20 = model.dna20
        self.dna21 = model.dna21
        self.dna22 = model.dna22
        self.dna23 = model.dna23
        self.dna24 = model.dna24
        self.dna25 = model.dna25
        self.dna26 = model.dna26
        self.dna27 = model.dna27
        self.dna28 = model.dna28
        self.dna29 = model.dna29
        self.dna30 = model.dna30
        self.dna31 = model.dna31
        self.dna32 = model.dna32
        return
    
    def create(self, width, height, depth, bounds, num_controls, num_sensations):    
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
        self.__check_cuda()
        self.width = width
        print('assigned width')
        self.height = height
        print('assigned height')
        self.depth = depth
        print('assigned depth')
        self.bounds = bounds
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
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.thresholds_pos = torch.tensor(data=1, device=self.device)
        self.thresholds_neg = torch.tensor(data=1, device=self.device)
        self.thresholds_pos = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=self.device)
        torch.add(torch.mul(self.thresholds_pos, self.bounds, out=self.thresholds_pos), 1, out=self.thresholds_pos)
        random_gen.seed()
        self.thresholds_neg = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=self.device)
        torch.subtract(-1, torch.divide(self.thresholds_neg, self.bounds, out=self.thresholds_neg), out=self.thresholds_neg)

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
        self.layer0 = torch.tensor(data=1, device=self.device)
        self.layer1 = torch.tensor(data=1, device=self.device)
        self.layer2 = torch.tensor(data=1, device=self.device)
        self.layer3 = torch.tensor(data=1, device=self.device)
        self.layer4 = torch.tensor(data=1, device=self.device)
        self.emotion1 = torch.tensor(data=1, device=self.device)
        self.emotion2 = torch.tensor(data=1, device=self.device)
        self.emotion3 = torch.tensor(data=1, device=self.device)
        self.emotion4 = torch.tensor(data=1, device=self.device)
        self.emotion5 = torch.tensor(data=1, device=self.device)
        self.emotion6 = torch.tensor(data=1, device=self.device)
        self.emotion7 = torch.tensor(data=1, device=self.device)
        self.emotion8 = torch.tensor(data=1, device=self.device)
        self.personality1 = torch.tensor(data=1, device=self.device)
        self.personality2 = torch.tensor(data=1, device=self.device)
        self.personality3 = torch.tensor(data=1, device=self.device)
        self.personality4 = torch.tensor(data=1, device=self.device)
        self.personality5 = torch.tensor(data=1, device=self.device)
        self.personality6 = torch.tensor(data=1, device=self.device)
        self.personality7 = torch.tensor(data=1, device=self.device)
        self.personality8 = torch.tensor(data=1, device=self.device)
        self.personality9 = torch.tensor(data=1, device=self.device)
        self.personality10 = torch.tensor(data=1, device=self.device)
        self.personality11 = torch.tensor(data=1, device=self.device)
        self.personality12 = torch.tensor(data=1, device=self.device)
        self.personality13 = torch.tensor(data=1, device=self.device)
        self.personality14 = torch.tensor(data=1, device=self.device)
        self.personality15 = torch.tensor(data=1, device=self.device)
        self.personality16 = torch.tensor(data=1, device=self.device)
        self.dna1 = torch.tensor(data=1, device=self.device)
        self.dna2 = torch.tensor(data=1, device=self.device)
        self.dna3 = torch.tensor(data=1, device=self.device)
        self.dna4 = torch.tensor(data=1, device=self.device)
        self.dna5 = torch.tensor(data=1, device=self.device)
        self.dna6 = torch.tensor(data=1, device=self.device)
        self.dna7 = torch.tensor(data=1, device=self.device)
        self.dna8 = torch.tensor(data=1, device=self.device)
        self.dna9 = torch.tensor(data=1, device=self.device)
        self.dna10 = torch.tensor(data=1, device=self.device)
        self.dna11 = torch.tensor(data=1, device=self.device)
        self.dna12 = torch.tensor(data=1, device=self.device)
        self.dna13 = torch.tensor(data=1, device=self.device)
        self.dna14 = torch.tensor(data=1, device=self.device)
        self.dna15 = torch.tensor(data=1, device=self.device)
        self.dna16 = torch.tensor(data=1, device=self.device)
        self.dna17 = torch.tensor(data=1, device=self.device)
        self.dna18 = torch.tensor(data=1, device=self.device)
        self.dna19 = torch.tensor(data=1, device=self.device)
        self.dna20 = torch.tensor(data=1, device=self.device)
        self.dna21 = torch.tensor(data=1, device=self.device)
        self.dna22 = torch.tensor(data=1, device=self.device)
        self.dna23 = torch.tensor(data=1, device=self.device)
        self.dna24 = torch.tensor(data=1, device=self.device)
        self.dna25 = torch.tensor(data=1, device=self.device)
        self.dna26 = torch.tensor(data=1, device=self.device)
        self.dna27 = torch.tensor(data=1, device=self.device)
        self.dna28 = torch.tensor(data=1, device=self.device)
        self.dna29 = torch.tensor(data=1, device=self.device)
        self.dna30 = torch.tensor(data=1, device=self.device)
        self.dna31 = torch.tensor(data=1, device=self.device)
        self.dna32 = torch.tensor(data=1, device=self.device)
        '''
        print(self.layer0)
        print(self.emotion1)
        print(self.personality1)
        '''
        # neuron layer
        self.layer0 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        # threshold layers
        self.layer1 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.layer2 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        #multiplier layers
        self.layer3 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.layer4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        # emotion layers
        self.emotion1 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion2 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion3 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion5 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion6 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion7 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.emotion8 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        
        # personality layers
        self.personality1 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality2 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality3 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality5 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality6 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality7 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality8 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality9 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality10 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality11 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality12 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality13 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality14 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality15 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.personality16 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16,device=self.device)
        
        #dna layers
        self.dna1 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna2 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna3 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna4 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna5 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna6 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna7 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna8 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna9 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna10 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna11 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna12 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna13 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna14 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna15 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna16 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna17 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna18 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna19 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna20 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna21 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna22 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna23 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna24 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna25 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna26 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna27 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna28 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna29 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna30 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna31 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.dna32 = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        
        #print(self.layer0)
        #print(self.emotion1)
        #print(self.personality1)
        self.positive_firing= torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.positive_resting = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.negative_firing = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.negative_resting = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.pos_fire_amt = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.neg_fire_amt = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.pos_fire_amt_mult = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        self.neg_fire_amt_mult = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        
        # personality layers
        # positive thresh firing is used
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.dna1 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna2 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna3 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna4 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna5 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna6 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna7 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna8 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna9 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna10 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna11 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna12 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna13 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna14 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna15 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna16 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna17 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna18 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna19 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna20 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna21 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna22 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna23 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna24 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna25 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna26 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna27 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna28 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna29 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna30 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna31 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)
        random_gen.seed()
        self.dna32 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device))).to(dtype=torch.int16)

        return
    
    def __new_propensity(self):
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.pos_propensity = torch.tensor(data=1, device=self.device)
        self.neg_propensity = torch.tensor(data=1, device=self.device)
        self.pos_propensity = torch.rand(size=(2,2), generator=random_gen, device=self.device)
        self.pos_propensity = torch.add(torch.mul(self.pos_propensity, self.bounds, out=self.pos_propensity), 1).to(dtype=torch.int16)
        self.pos_propensity = torch.subtract(self.pos_propensity, torch.divide(self.pos_propensity, 2).to(dtype=torch.int16), out=self.pos_propensity)
        self.neg_propensity = torch.clone(self.pos_propensity)
        #nep = torch.rand(size=(2,2), generator=random_gen, dtype=torch.int16, device=self.device)
        #pep = torch.ones(size=(2,2), device=self.device)
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
        model = general_dev2()
        model.copy(self)
        model.__new_thresholds()
        model.__new_propensity()
        model.__new_personality()
        torch.divide(model.thresholds_pos, fraction, out=model.thresholds_pos)
        torch.divide(model.thresholds_neg, fraction, out=model.thresholds_neg)
        model.dna1 = torch.divide(model.dna1, fraction,).to(dtype=torch.int16)
        model.dna2 = torch.divide(model.dna2, fraction,).to(dtype=torch.int16)
        model.dna3 = torch.divide(model.dna3, fraction,).to(dtype=torch.int16)
        model.dna4 = torch.divide(model.dna4, fraction,).to(dtype=torch.int16)
        model.dna5 = torch.divide(model.dna5, fraction,).to(dtype=torch.int16)
        model.dna6 = torch.divide(model.dna6, fraction,).to(dtype=torch.int16)
        model.dna7 = torch.divide(model.dna7, fraction,).to(dtype=torch.int16)
        model.dna8 = torch.divide(model.dna8, fraction,).to(dtype=torch.int16)
        model.dna9 = torch.divide(model.dna9, fraction,).to(dtype=torch.int16)
        model.dna10 = torch.divide(model.dna10, fraction,).to(dtype=torch.int16)
        model.dna11 = torch.divide(model.dna11, fraction,).to(dtype=torch.int16)
        model.dna12 = torch.divide(model.dna12, fraction,).to(dtype=torch.int16)
        model.dna13 = torch.divide(model.dna13, fraction,).to(dtype=torch.int16)
        model.dna14 = torch.divide(model.dna14, fraction,).to(dtype=torch.int16)
        model.dna15 = torch.divide(model.dna15, fraction,).to(dtype=torch.int16)
        model.dna16 = torch.divide(model.dna16, fraction,).to(dtype=torch.int16)
        model.dna17 = torch.divide(model.dna17, fraction,).to(dtype=torch.int16)
        model.dna18 = torch.divide(model.dna18, fraction,).to(dtype=torch.int16)
        model.dna19 = torch.divide(model.dna19, fraction,).to(dtype=torch.int16)
        model.dna20 = torch.divide(model.dna20, fraction,).to(dtype=torch.int16)
        model.dna21 = torch.divide(model.dna21, fraction,).to(dtype=torch.int16)
        model.dna22 = torch.divide(model.dna22, fraction,).to(dtype=torch.int16)
        model.dna23 = torch.divide(model.dna23, fraction,).to(dtype=torch.int16)
        model.dna24 = torch.divide(model.dna24, fraction,).to(dtype=torch.int16)
        model.dna25 = torch.divide(model.dna25, fraction,).to(dtype=torch.int16)
        model.dna26 = torch.divide(model.dna26, fraction,).to(dtype=torch.int16)
        model.dna27 = torch.divide(model.dna27, fraction,).to(dtype=torch.int16)
        model.dna28 = torch.divide(model.dna28, fraction,).to(dtype=torch.int16)
        model.dna29 = torch.divide(model.dna29, fraction,).to(dtype=torch.int16)
        model.dna30 = torch.divide(model.dna30, fraction,).to(dtype=torch.int16)
        model.dna31 = torch.divide(model.dna31, fraction,).to(dtype=torch.int16)
        model.dna32 = torch.divide(model.dna32, fraction,).to(dtype=torch.int16)
        for i in range(0, degree):
            torch.add(self.thresholds_pos, model.thresholds_pos, out=self.thresholds_pos)
            torch.add(self.thresholds_neg, model.thresholds_neg, out=self.thresholds_neg)
            torch.add(self.dna1, model.dna1, out=self.dna1)
            torch.add(self.dna2, model.dna2, out=self.dna2)
            torch.add(self.dna3, model.dna3, out=self.dna3)
            torch.add(self.dna4, model.dna4, out=self.dna4)
            torch.add(self.dna5, model.dna5, out=self.dna5)
            torch.add(self.dna6, model.dna6, out=self.dna6)
            torch.add(self.dna7, model.dna7, out=self.dna7)
            torch.add(self.dna8, model.dna8, out=self.dna8)
            torch.add(self.dna9, model.dna9, out=self.dna9)
            torch.add(self.dna10, model.dna10, out=self.dna10)
            torch.add(self.dna11, model.dna11, out=self.dna11)
            torch.add(self.dna12, model.dna12, out=self.dna12)
            torch.add(self.dna13, model.dna13, out=self.dna13)
            torch.add(self.dna14, model.dna14, out=self.dna14)
            torch.add(self.dna15, model.dna15, out=self.dna15)
            torch.add(self.dna16, model.dna16, out=self.dna16)
            torch.add(self.dna17, model.dna17, out=self.dna17)
            torch.add(self.dna18, model.dna18, out=self.dna18)
            torch.add(self.dna19, model.dna19, out=self.dna19)
            torch.add(self.dna20, model.dna20, out=self.dna20)
            torch.add(self.dna21, model.dna21, out=self.dna21)
            torch.add(self.dna22, model.dna22, out=self.dna22)
            torch.add(self.dna23, model.dna23, out=self.dna23)
            torch.add(self.dna24, model.dna24, out=self.dna24)
            torch.add(self.dna25, model.dna25, out=self.dna25)
            torch.add(self.dna26, model.dna26, out=self.dna26)
            torch.add(self.dna27, model.dna27, out=self.dna27)
            torch.add(self.dna28, model.dna28, out=self.dna28)
            torch.add(self.dna29, model.dna29, out=self.dna29)
            torch.add(self.dna30, model.dna30, out=self.dna30)
            torch.add(self.dna31, model.dna31, out=self.dna31)
            torch.add(self.dna32, model.dna32, out=self.dna32)
        deg = 1 + (degree / fraction)
        torch.divide(self.thresholds_pos, deg + 1, out=self.thresholds_pos)
        torch.divide(self.thresholds_neg, deg + 1, out=self.thresholds_neg)
        self.dna1 = torch.divide(self.dna1, deg,).to(dtype=torch.int16)
        self.dna2 = torch.divide(self.dna2, deg,).to(dtype=torch.int16)
        self.dna3 = torch.divide(self.dna3, deg,).to(dtype=torch.int16)
        self.dna4 = torch.divide(self.dna4, deg,).to(dtype=torch.int16)
        self.dna5 = torch.divide(self.dna5, deg,).to(dtype=torch.int16)
        self.dna6 = torch.divide(self.dna6, deg,).to(dtype=torch.int16)
        self.dna7 = torch.divide(self.dna7, deg,).to(dtype=torch.int16)
        self.dna8 = torch.divide(self.dna8, deg,).to(dtype=torch.int16)
        self.dna9 = torch.divide(self.dna9, deg,).to(dtype=torch.int16)
        self.dna10 = torch.divide(self.dna10, deg,).to(dtype=torch.int16)
        self.dna11 = torch.divide(self.dna11, deg,).to(dtype=torch.int16)
        self.dna12 = torch.divide(self.dna12, deg,).to(dtype=torch.int16)
        self.dna13 = torch.divide(self.dna13, deg,).to(dtype=torch.int16)
        self.dna14 = torch.divide(self.dna14, deg,).to(dtype=torch.int16)
        self.dna15 = torch.divide(self.dna15, deg,).to(dtype=torch.int16)
        self.dna16 = torch.divide(self.dna16, deg,).to(dtype=torch.int16)
        self.dna17 = torch.divide(self.dna17, deg,).to(dtype=torch.int16)
        self.dna18 = torch.divide(self.dna18, deg,).to(dtype=torch.int16)
        self.dna19 = torch.divide(self.dna19, deg,).to(dtype=torch.int16)
        self.dna20 = torch.divide(self.dna20, deg,).to(dtype=torch.int16)
        self.dna21 = torch.divide(self.dna21, deg,).to(dtype=torch.int16)
        self.dna22 = torch.divide(self.dna22, deg,).to(dtype=torch.int16)
        self.dna23 = torch.divide(self.dna23, deg,).to(dtype=torch.int16)
        self.dna24 = torch.divide(self.dna24, deg,).to(dtype=torch.int16)
        self.dna25 = torch.divide(self.dna25, deg,).to(dtype=torch.int16)
        self.dna26 = torch.divide(self.dna26, deg,).to(dtype=torch.int16)
        self.dna27 = torch.divide(self.dna27, deg,).to(dtype=torch.int16)
        self.dna28 = torch.divide(self.dna28, deg,).to(dtype=torch.int16)
        self.dna29 = torch.divide(self.dna29, deg,).to(dtype=torch.int16)
        self.dna30 = torch.divide(self.dna30, deg,).to(dtype=torch.int16)
        self.dna31 = torch.divide(self.dna31, deg,).to(dtype=torch.int16)
        self.dna32 = torch.divide(self.dna32, deg,).to(dtype=torch.int16)
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
        input_image.to(dtype=torch.int16, device=self.device)
        input_tensor = torch.tensor(data=1, device=self.device)
        
        input_tensor = torch.clone(input_image, ).detach()
        #print(input_image.device)
        #print(input_tensor)
        #torch.div(input_tensor, 255, out=input_tensor)
        #torch.mul(input_tensor, self.bounds, out=input_tensor)
        #torch.add(input_tensor, torch.ones(size=input_image.size(), device=self.device), out=input_tensor)
        #print(input_tensor)
        #print('layer0')
        #print(self.layer0)o
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
        pos_firing_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.int16)
        pos_resting_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.int16)
        neg_firing_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.int16)
        neg_resting_NOT = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.int16)
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
        temp = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.int16)
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
            if (self.layer0[self.controls[i][0], self.controls[i][1], self.controls[i][2]].item() > self.thresholds_pos[i, 0].item()):
                take_action.append(True)
                self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])] = self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])].item() - self.thresholds_pos[i,0]
            else:
                if (self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])].item() > self.thresholds_neg[i,0].item()):
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
        
        # now update the personality values according to their associated dna values
        
        torch.add(torch.mul(self.positive_firing, self.dna1), self.personality1, out=self.personality1)
        torch.add(torch.mul(self.positive_resting, self.dna2), self.personality1, out=self.personality1)
        torch.add(torch.mul(self.negative_firing, self.dna3), self.personality2, out=self.personality2)
        torch.add(torch.mul(self.negative_resting, self.dna4), self.personality2, out=self.personality2)
        torch.add(torch.mul(self.positive_firing, self.dna5), self.personality3, out=self.personality3)
        torch.add(torch.mul(self.positive_resting, self.dna6), self.personality3, out=self.personality3)
        torch.add(torch.mul(self.negative_firing, self.dna7), self.personality4, out=self.personality4)
        torch.add(torch.mul(self.negative_resting, self.dna8), self.personality4, out=self.personality4)
        torch.add(torch.mul(self.positive_firing, self.dna9), self.personality5, out=self.personality5)
        torch.add(torch.mul(self.positive_resting, self.dna10), self.personality5, out=self.personality5)
        torch.add(torch.mul(self.negative_firing, self.dna11), self.personality6, out=self.personality6)
        torch.add(torch.mul(self.negative_resting, self.dna12), self.personality6, out=self.personality6)
        torch.add(torch.mul(self.positive_firing, self.dna13), self.personality7, out=self.personality7)
        torch.add(torch.mul(self.positive_resting, self.dna14), self.personality7, out=self.personality7)
        torch.add(torch.mul(self.negative_firing, self.dna15), self.personality8, out=self.personality8)
        torch.add(torch.mul(self.negative_resting, self.dna16), self.personality8, out=self.personality8)
        torch.add(torch.mul(self.positive_firing, self.dna17), self.personality9, out=self.personality9)
        torch.add(torch.mul(self.positive_resting, self.dna18), self.personality9, out=self.personality9)
        torch.add(torch.mul(self.negative_firing, self.dna19), self.personality10, out=self.personality10)
        torch.add(torch.mul(self.negative_resting, self.dna20), self.personality10, out=self.personality10)
        torch.add(torch.mul(self.positive_firing, self.dna21), self.personality11, out=self.personality11)
        torch.add(torch.mul(self.positive_resting, self.dna22), self.personality11, out=self.personality11)
        torch.add(torch.mul(self.negative_firing, self.dna23), self.personality12, out=self.personality12)
        torch.add(torch.mul(self.negative_resting, self.dna24), self.personality12, out=self.personality12)
        torch.add(torch.mul(self.positive_firing, self.dna25), self.personality13, out=self.personality13)
        torch.add(torch.mul(self.positive_resting, self.dna26), self.personality13, out=self.personality13)
        torch.add(torch.mul(self.negative_firing, self.dna27), self.personality14, out=self.personality14)
        torch.add(torch.mul(self.negative_resting, self.dna28), self.personality14, out=self.personality14)
        torch.add(torch.mul(self.positive_firing, self.dna29), self.personality15, out=self.personality15)
        torch.add(torch.mul(self.positive_resting, self.dna30), self.personality15, out=self.personality15)
        torch.add(torch.mul(self.negative_firing, self.dna31), self.personality16, out=self.personality16)
        torch.add(torch.mul(self.negative_resting, self.dna32), self.personality16, out=self.personality16)

        #print(take_action)
        return take_action




class velvet():
    # variables and definitions
    device = -1
    min_dx = -1
    min_dy = -1
    width = 0
    height = 0
    depth = 0
    bounds = 0
    colors = 255
    num_controls = 0
    controls = []
    control_thresholds_pos = []
    control_thresholds_neg = []
    num_sensations = 0
    sensations = []
    pos_propensity  = 0
    neg_propensity = 0

    propensity = .001
    
    # layers:
    # neuron layer
    layer0 = 0
    
    # threshold layers
    layer1 = 0
    layer2 = 0
    
    # signal layers
    layer3 = 0
    layer4 = 0
    
    # emotion layers
    layer5 = 0
    layer6 = 0
    layer7 = 0
    layer8 = 0
    layer9 = 0
    layer10 = 0
    layer11 = 0
    layer12 = 0

    # personality layers
    layer13 = 0
    layer14 = 0
    layer15 = 0
    layer16 = 0
    layer17 = 0
    layer18 = 0
    layer19 = 0
    layer20 = 0
    layer21 = 0
    layer22 = 0
    layer23 = 0
    layer24 = 0
    layer25 = 0
    layer26 = 0
    layer27 = 0
    layer28 = 0


    # dna layers
    layer29 = 0
    layer30 = 0
    layer31 = 0
    layer32 = 0
    layer33 = 0
    layer34 = 0
    layer35 = 0
    layer36 = 0
    layer37 = 0
    layer38 = 0
    layer39 = 0
    layer40 = 0
    layer41 = 0
    layer42 = 0
    layer43 = 0
    layer44 = 0
    layer45 = 0
    layer46 = 0
    layer47 = 0
    layer48 = 0
    layer49 = 0
    layer50 = 0
    layer51 = 0
    layer52 = 0
    layer53 = 0
    layer54 = 0
    layer55 = 0
    layer56 = 0
    layer57 = 0
    layer58 = 0
    layer59 = 0
    layer60 = 0

    def __check_cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return


    def __init__(self):
        self.__check_cuda()
        return

    def create(self, width, height, depth, bounds, num_controls, num_sensations):
        self.__check_cuda()
        self.width = width
        print('assigned width')
        self.height = height
        print('assigned height')
        self.depth = depth
        print('assigned depth')
        self.bounds = bounds
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
        self.__new_dna()
        print('new personality')
        self.__new_sensations()
        print('new sensations')
        return

    def __new_controls(self):
        self.controls = []
        for i in range(0, self.num_controls):
            wegood = False
            newctl = 0
            while wegood == False:
                newctl = (np.random.randint(low=7, high=self.width), np.random.randint(low=1, high=self.height), self.depth)
                wegood = True
                for ctl in self.controls:
                    if ctl == newctl:
                        wegood = False
            self.controls.append(newctl)
        return

    def __new_propensity(self):
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.pos_propensity = torch.tensor(data=1, device=self.device)
        self.neg_propensity = torch.tensor(data=1, device=self.device)
        self.pos_propensity = torch.rand(size=(2,2), generator=random_gen, device=self.device)
        self.pos_propensity = torch.add(torch.mul(self.pos_propensity, self.bounds, out=self.pos_propensity), 1)
        self.pos_propensity = torch.subtract(self.pos_propensity, torch.divide(self.pos_propensity, 2), out=self.pos_propensity)
        self.neg_propensity = torch.clone(self.pos_propensity)
        return
    
    def __new_dna(self):
        self.layer0 = torch.tensor(data=1, device=self.device)
        self.layer1 = torch.tensor(data=1, device=self.device)
        self.layer2 = torch.tensor(data=1, device=self.device)
        self.layer3 = torch.tensor(data=1, device=self.device)
        self.layer4 = torch.tensor(data=1, device=self.device)
        self.layer5 = torch.tensor(data=1, device=self.device)
        self.layer6 = torch.tensor(data=1, device=self.device)
        self.layer7 = torch.tensor(data=1, device=self.device)
        self.layer8 = torch.tensor(data=1, device=self.device)
        self.layer9 = torch.tensor(data=1, device=self.device)
        self.layer10 = torch.tensor(data=1, device=self.device)
        self.layer11 = torch.tensor(data=1, device=self.device)
        self.layer12 = torch.tensor(data=1, device=self.device)
        self.layer13 = torch.tensor(data=1, device=self.device)
        self.layer14 = torch.tensor(data=1, device=self.device)
        self.layer15 = torch.tensor(data=1, device=self.device)
        self.layer16 = torch.tensor(data=1, device=self.device)
        self.layer17 = torch.tensor(data=1, device=self.device)
        self.layer18 = torch.tensor(data=1, device=self.device)
        self.layer19 = torch.tensor(data=1, device=self.device)
        self.layer20 = torch.tensor(data=1, device=self.device)
        self.layer21 = torch.tensor(data=1, device=self.device)
        self.layer22 = torch.tensor(data=1, device=self.device)
        self.layer23 = torch.tensor(data=1, device=self.device)
        self.layer24 = torch.tensor(data=1, device=self.device)
        self.layer25 = torch.tensor(data=1, device=self.device)
        self.layer26 = torch.tensor(data=1, device=self.device)
        self.layer27 = torch.tensor(data=1, device=self.device)
        self.layer28 = torch.tensor(data=1, device=self.device)
        self.layer29 = torch.tensor(data=1, device=self.device)
        self.layer30 = torch.tensor(data=1, device=self.device)
        self.layer31 = torch.tensor(data=1, device=self.device)
        self.layer32 = torch.tensor(data=1, device=self.device)
        self.layer33 = torch.tensor(data=1, device=self.device)
        self.layer34 = torch.tensor(data=1, device=self.device)
        self.layer35 = torch.tensor(data=1, device=self.device)
        self.layer36 = torch.tensor(data=1, device=self.device)
        self.layer37 = torch.tensor(data=1, device=self.device)
        self.layer38 = torch.tensor(data=1, device=self.device)
        self.layer39 = torch.tensor(data=1, device=self.device)
        self.layer40 = torch.tensor(data=1, device=self.device)
        self.layer41 = torch.tensor(data=1, device=self.device)
        self.layer42 = torch.tensor(data=1, device=self.device)
        self.layer43 = torch.tensor(data=1, device=self.device)
        self.layer44 = torch.tensor(data=1, device=self.device)
        self.layer45 = torch.tensor(data=1, device=self.device)
        self.layer46 = torch.tensor(data=1, device=self.device)
        self.layer47 = torch.tensor(data=1, device=self.device)
        self.layer48 = torch.tensor(data=1, device=self.device)
        self.layer49 = torch.tensor(data=1, device=self.device)
        self.layer50 = torch.tensor(data=1, device=self.device)
        self.layer51 = torch.tensor(data=1, device=self.device)
        self.layer52 = torch.tensor(data=1, device=self.device)
        self.layer53 = torch.tensor(data=1, device=self.device)
        self.layer54 = torch.tensor(data=1, device=self.device)
        self.layer55 = torch.tensor(data=1, device=self.device)
        self.layer56 = torch.tensor(data=1, device=self.device)
        self.layer57 = torch.tensor(data=1, device=self.device)
        self.layer58 = torch.tensor(data=1, device=self.device)
        self.layer59 = torch.tensor(data=1, device=self.device)
        self.layer60 = torch.tensor(data=1, device=self.device)
        
        self.layer0 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer1 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer2 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer3 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer4 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer5 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer6 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer7 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer8 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer9 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer10 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer11 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer12 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer13 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer14 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer15 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer16 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer17 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer18 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer19 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer20 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer21 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer22 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer23 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer24 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer25 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer26 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer27 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer28 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer29 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer30 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer31 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer32 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer33 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer34 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer35 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer36 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer37 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer38 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer39 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer40 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer41 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer42 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer43 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer44 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer45 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer46 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer47 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer48 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer49 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer50 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer51 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer52 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer53 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer54 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer55 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer56 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer57 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer58 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer59 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer60 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.layer29 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer30 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer31 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer32 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer33 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer34 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer35 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer36 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer37 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer38 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer39 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer40 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer41 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer42 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer43 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer44 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer45 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer46 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer47 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer48 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer49 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer50 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer51 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer52 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer53 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer54 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer55 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer56 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer57 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer58 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer59 = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))
        random_gen.seed()
        self.layer60 = torch.multiply(other=self.neg_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)))

    def save(self, path):
        np.save(path + '/width', self.width)
        np.save(path + '/height', self.height)
        np.save(path + '/depth', self.depth)
        np.save(path + '/bounds', self.bounds)
        np.save(path + '/num_controls', self.num_controls) 
        np.save(path + '/controls', self.controls)
        np.save(path + '/num_sensations', self.num_sensations)
        np.save(path + '/sensations', self.sensations)
        torch.save(self.control_thresholds_pos, path + '/control_thresholds_pos.pth')
        torch.save(self.control_thresholds_neg, path + '/control_thresholds_neg.pth')
        torch.save(self.layer0, path + '/layer0.pth')
        torch.save(self.layer1, path + '/layer1.pth')
        torch.save(self.layer2, path + '/layer2.pth')
        torch.save(self.layer3, path + '/layer3.pth')
        torch.save(self.layer4, path + '/layer4.pth')
        torch.save(self.layer5, path + '/layer5.pth')
        torch.save(self.layer6, path + '/layer6.pth')
        torch.save(self.layer7, path + '/layer7.pth')
        torch.save(self.layer8, path + '/layer8.pth')
        torch.save(self.layer9, path + '/layer9.pth')
        torch.save(self.layer10, path + '/layer10.pth')
        torch.save(self.layer11, path + '/layer11.pth')
        torch.save(self.layer12, path + '/layer12.pth')
        torch.save(self.layer13, path + '/layer13.pth')
        torch.save(self.layer14, path + '/layer14.pth')
        torch.save(self.layer15, path + '/layer15.pth')
        torch.save(self.layer16, path + '/layer16.pth')
        torch.save(self.layer17, path + '/layer17.pth')
        torch.save(self.layer18, path + '/layer18.pth')
        torch.save(self.layer19, path + '/layer19.pth')
        torch.save(self.layer20, path + '/layer20.pth')
        torch.save(self.layer21, path + '/layer21.pth')
        torch.save(self.layer22, path + '/layer22.pth')
        torch.save(self.layer23, path + '/layer23.pth')
        torch.save(self.layer24, path + '/layer24.pth')
        torch.save(self.layer25, path + '/layer25.pth')
        torch.save(self.layer26, path + '/layer26.pth')
        torch.save(self.layer27, path + '/layer27.pth')
        torch.save(self.layer28, path + '/layer28.pth')
        torch.save(self.layer29, path + '/layer29.pth')
        torch.save(self.layer30, path + '/layer30.pth')
        torch.save(self.layer31, path + '/layer31.pth')
        torch.save(self.layer32, path + '/layer32.pth')
        torch.save(self.layer33, path + '/layer33.pth')
        torch.save(self.layer34, path + '/layer34.pth')
        torch.save(self.layer35, path + '/layer35.pth')
        torch.save(self.layer36, path + '/layer36.pth')
        torch.save(self.layer37, path + '/layer37.pth')
        torch.save(self.layer38, path + '/layer38.pth')
        torch.save(self.layer39, path + '/layer39.pth')
        torch.save(self.layer40, path + '/layer40.pth')
        torch.save(self.layer41, path + '/layer41.pth')
        torch.save(self.layer42, path + '/layer42.pth')
        torch.save(self.layer43, path + '/layer43.pth')
        torch.save(self.layer44, path + '/layer44.pth')
        torch.save(self.layer45, path + '/layer45.pth')
        torch.save(self.layer46, path + '/layer46.pth')
        torch.save(self.layer47, path + '/layer47.pth')
        torch.save(self.layer48, path + '/layer48.pth')
        torch.save(self.layer49, path + '/layer49.pth')
        torch.save(self.layer50, path + '/layer50.pth')
        torch.save(self.layer51, path + '/layer51.pth')
        torch.save(self.layer52, path + '/layer52.pth')
        torch.save(self.layer53, path + '/layer53.pth')
        torch.save(self.layer54, path + '/layer54.pth')
        torch.save(self.layer55, path + '/layer55.pth')
        torch.save(self.layer56, path + '/layer56.pth')
        torch.save(self.layer57, path + '/layer57.pth')
        torch.save(self.layer58, path + '/layer58.pth')
        torch.save(self.layer59, path + '/layer59.pth')
        torch.save(self.layer60, path + '/layer60.pth')
        return

    def load(self, path):
        self.__check_cuda()
        self.width = np.load(path + '/width.npy')
        self.height = np.load(path + '/height.npy')
        self.depth = np.load(path + '/depth.npy')
        self.bounds = np.load(path + '/bounds.npy')
        self.num_controls = np.load(path + '/num_controls.npy')
        self.controls = np.load(path + '/controls.npy')
        self.num_sensations = np.load(path + '/num_sensations.npy')
        self.sensations = np.load(path + '/sensations.npy')
        self.control_thresholds_pos = torch.load(path + '/thresholds_pos.pth')
        self.control_thresholds_neg = torch.load(path + '/thresholds_neg.pth')
        self.layer0 = torch.load(path + '/layer0.pth')
        self.layer1 = torch.load(path + '/layer1.pth')
        self.layer2 = torch.load(path + '/layer2.pth')
        self.layer3 = torch.load(path + '/layer3.pth')
        self.layer4 = torch.load(path + '/layer4.pth')
        self.layer5 = torch.load(path + '/layer5.pth')
        self.layer6 = torch.load(path + '/layer6.pth')
        self.layer7 = torch.load(path + '/layer7.pth')
        self.layer8 = torch.load(path + '/layer8.pth')
        self.layer9 = torch.load(path + '/layer9.pth')
        self.layer10 = torch.load(path + '/layer10.pth')
        self.layer11 = torch.load(path + '/layer11.pth')
        self.layer12 = torch.load(path + '/layer12.pth')
        self.layer13 = torch.load(path + '/layer13.pth')
        self.layer14 = torch.load(path + '/layer14.pth')
        self.layer15 = torch.load(path + '/layer15.pth')
        self.layer16 = torch.load(path + '/layer16.pth')
        self.layer17 = torch.load(path + '/layer17.pth')
        self.layer18 = torch.load(path + '/layer18.pth')
        self.layer19 = torch.load(path + '/layer19.pth')
        self.layer20 = torch.load(path + '/layer20.pth')
        self.layer21 = torch.load(path + '/layer21.pth')
        self.layer22 = torch.load(path + '/layer22.pth')
        self.layer23 = torch.load(path + '/layer23.pth')
        self.layer24 = torch.load(path + '/layer24.pth')
        self.layer25 = torch.load(path + '/layer25.pth')
        self.layer26 = torch.load(path + '/layer26.pth')
        self.layer27 = torch.load(path + '/layer27.pth')
        self.layer28 = torch.load(path + '/layer28.pth')
        self.layer29 = torch.load(path + '/layer29.pth')
        self.layer30 = torch.load(path + '/layer30.pth')
        self.layer31 = torch.load(path + '/layer31.pth')
        self.layer32 = torch.load(path + '/layer32.pth')
        self.layer33 = torch.load(path + '/layer33.pth')
        self.layer34 = torch.load(path + '/layer34.pth')
        self.layer35 = torch.load(path + '/layer35.pth')
        self.layer36 = torch.load(path + '/layer36.pth')
        self.layer37 = torch.load(path + '/layer37.pth')
        self.layer38 = torch.load(path + '/layer38.pth')
        self.layer39 = torch.load(path + '/layer39.pth')
        self.layer40 = torch.load(path + '/layer40.pth')
        self.layer41 = torch.load(path + '/layer41.pth')
        self.layer42 = torch.load(path + '/layer42.pth')
        self.layer43 = torch.load(path + '/layer43.pth')
        self.layer44 = torch.load(path + '/layer44.pth')
        self.layer45 = torch.load(path + '/layer45.pth')
        self.layer46 = torch.load(path + '/layer46.pth')
        self.layer47 = torch.load(path + '/layer47.pth')
        self.layer48 = torch.load(path + '/layer48.pth')
        self.layer49 = torch.load(path + '/layer49.pth')
        self.layer50 = torch.load(path + '/layer50.pth')
        self.layer51 = torch.load(path + '/layer51.pth')
        self.layer52 = torch.load(path + '/layer52.pth')
        self.layer53 = torch.load(path + '/layer53.pth')
        self.layer54 = torch.load(path + '/layer54.pth')
        self.layer55 = torch.load(path + '/layer55.pth')
        self.layer56 = torch.load(path + '/layer56.pth')
        self.layer57 = torch.load(path + '/layer57.pth')
        self.layer58 = torch.load(path + '/layer58.pth')
        self.layer59 = torch.load(path + '/layer59.pth')
        self.layer60 = torch.load(path + '/layer60.pth')
        return
        
    def __new_thresholds(self):
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.thresholds_pos = torch.tensor(data=1, device=self.device)
        self.thresholds_neg = torch.tensor(data=1, device=self.device)
        self.thresholds_pos = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=self.device)
        torch.add(torch.mul(self.thresholds_pos, self.bounds, out=self.thresholds_pos), 1, out=self.thresholds_pos)
        random_gen.seed()
        self.thresholds_neg = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=self.device)
        torch.subtract(-1, torch.divide(self.thresholds_neg, self.bounds, out=self.thresholds_neg), out=self.thresholds_neg)
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

    def copy(self, model):
        '''
        Copy a model's parameters to a new model.
        
        :Parameters:
        model (general_dev2): the model to copy
        
        :Returns:
        none
        '''
        self.width = model.width
        self.height = model.height
        self.depth = model.depth
        self.bounds = model.bounds
        self.num_controls = model.num_controls
        self.controls = model.controls
        self.thresholds_pos = model.thresholds_pos
        self.thresholds_neg = model.thresholds_neg
        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.layer5 = model.layer5
        self.layer6 = model.layer6
        self.layer7 = model.layer7
        self.layer8 = model.layer8
        self.layer9 = model.layer9
        self.layer10 = model.layer10
        self.layer11 = model.layer11
        self.layer12 = model.layer12
        self.layer13 = model.layer13
        self.layer14 = model.layer14
        self.layer15 = model.layer15
        self.layer16 = model.layer16
        self.layer17 = model.layer17
        self.layer18 = model.layer18
        self.layer19 = model.layer19
        self.layer20 = model.layer20
        self.layer21 = model.layer21
        self.layer22 = model.layer22
        self.layer23 = model.layer23
        self.layer24 = model.layer24
        self.layer25 = model.layer25
        self.layer26 = model.layer26
        self.layer27 = model.layer27
        self.layer28 = model.layer28
        self.layer29 = model.layer29
        self.layer30 = model.layer30
        self.layer31 = model.layer31
        self.layer32 = model.layer32
        self.layer33 = model.layer33
        self.layer34 = model.layer34
        self.layer35 = model.layer35
        self.layer36 = model.layer36
        self.layer37 = model.layer37
        self.layer38 = model.layer38
        self.layer39 = model.layer39
        self.layer40 = model.layer40
        self.layer41 = model.layer41
        self.layer42 = model.layer42
        self.layer43 = model.layer43
        self.layer44 = model.layer44
        self.layer45 = model.layer45
        self.layer46 = model.layer46
        self.layer47 = model.layer47
        self.layer48 = model.layer48
        self.layer49 = model.layer49
        self.layer50 = model.layer50
        self.layer51 = model.layer51
        self.layer52 = model.layer52
        self.layer53 = model.layer53
        self.layer54 = model.layer54
        self.layer55 = model.layer55
        self.layer56 = model.layer56
        self.layer57 = model.layer57
        self.layer58 = model.layer58
        self.layer59 = model.layer59
        self.layer60 = model.layer60
        return

    def clear(self):
        self.layer0 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer1 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer2 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer3 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer4 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer5 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer6 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer7 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer8 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer9 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer10 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer11 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer12 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer13 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer14 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer15 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer16 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer17 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer18 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer19 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer20 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer21 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer22 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer23 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer24 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer25 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer26 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer27 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer28 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)

    
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

    def permute(self, fraction):        
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

        threshp = self.thresholds_pos
        threshn = self.thresholds_neg
        self.__new_thresholds()
        torch.divide(self.thresholds_pos, fraction, out=self.thresholds_pos)
        torch.divide(self.thresholds_neg, fraction, out=self.thresholds_neg)
        torch.add(self.thresholds_pos, threshp, out=self.thresholds_pos)
        torch.add(self.thresholds_neg, threshn, out=self.thresholds_neg)

        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer29, temp, out=self.layer29)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer30, temp, out=self.layer30)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer31, temp, out=self.layer31)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer32, temp, out=self.layer32)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer33, temp, out=self.layer33)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer34, temp, out=self.layer34)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer35, temp, out=self.layer35)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer36, temp, out=self.layer36)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer37, temp, out=self.layer37)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer38, temp, out=self.layer38)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer39, temp, out=self.layer39)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer40, temp, out=self.layer40)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer41, temp, out=self.layer41)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer42, temp, out=self.layer42)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer43, temp, out=self.layer43)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer44, temp, out=self.layer44)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer45, temp, out=self.layer45)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer46, temp, out=self.layer46)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer47, temp, out=self.layer47)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer48, temp, out=self.layer48)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer49, temp, out=self.layer49)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer50, temp, out=self.layer50)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer50, temp, out=self.layer50)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer52, temp, out=self.layer52)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer53, temp, out=self.layer53)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer54, temp, out=self.layer54)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer55, temp, out=self.layer55)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer56, temp, out=self.layer56)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer57, temp, out=self.layer57)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer58, temp, out=self.layer58)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer59, temp, out=self.layer59)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer60, temp, out=self.layer60)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer60, temp, out=self.layer60)
        temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float32, device=self.device)), out=temp)
        torch.divide(temp, fraction, out=temp)
        torch.add(self.layer60, temp, out=self.layer60)
        
        self.layer0 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer1 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer2 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer3 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer4 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer5 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer6 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer7 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer8 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer9 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer10 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer11 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer12 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer13 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer14 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer15 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer16 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer17 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer18 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer19 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer20 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer21 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer22 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer23 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer24 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer25 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer26 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer27 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        self.layer28 = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float32, device=self.device)
        return
        
    def update(self, input_image):
        if (torch.is_tensor(input_image) == False):
            return -1
        # add in the input image
        input_image.to(dtype=torch.int16, device=self.device)
        input_tensor = torch.tensor(data=1, device=self.device)
        
        input_tensor = torch.clone(input_image, ).detach()
        #print(input_image.device)
        #print(input_tensor)
        #torch.div(input_tensor, 255, out=input_tensor)
        #torch.mul(input_tensor, self.bounds, out=input_tensor)
        #torch.add(input_tensor, torch.ones(size=input_image.size(), device=self.device), out=input_tensor)
        #print(input_tensor)
        #print('layer0')
        #print(self.layer0)o
        try:
            torch.add(self.layer0[:, :, 0],  input_tensor[0, :, :], out=self.layer0[:, :, 0])
            torch.add(self.layer0[:, :, 1],  input_tensor[1, :, :], out=self.layer0[:, :, 1])
            torch.add(self.layer0[:, :, 2],  input_tensor[2, :, :], out=self.layer0[:, :, 2])
        except:
            torch.add(self.layer0[:, :, 0],  input_tensor[0, :, :], out=self.layer0[:, :, 0])

        # update layer0 based on the arctan function we're using, as well as inputs from the threshold and signal layers
        torch.add(torch.mul(torch.mul(torch.atan(self.layer0), self.layer1), self.layer3), torch.mul(torch.mul(torch.atan(self.layer0), self.layer2), self.layer4))

        # do some rolls to simulate neurons sending messages to each other
        '''
        temp = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.float32)
        torch.mul(self.layer0, torch.roll(self.layer0, 1, 0), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, -1, 0), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, 1, 1), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, -1, 1), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, 1, 2), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, -1, 2), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, 1, 0), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, -1, 0), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, 1, 1), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, -1, 1), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, 1, 2), out=temp)
        torch.mul(self.layer0, torch.roll(self.layer0, -1, 2), out=temp)
        torch.mul(self.layer0, temp, out=self.layer0)
        '''
        '''print("3")
        print('layer0')
        print(self.layer0)'''
        #print(self.layer0[:,:,3].shape())

        # guess we're the outer sums factory now
        self.layer0[:,:,3] = torch.sum(torch.kron(self.layer0[:,:,0], self.layer0[:,:,3]), (-1, -2), True)
        self.layer0[:,:,4] = torch.sum(torch.kron(self.layer0[:,:,1], self.layer0[:,:,4]), (-1, -2), True)
        self.layer0[:,:,5] = torch.sum(torch.kron(self.layer0[:,:,2], self.layer0[:,:,5]), (-1, -2), True)
        self.layer0[:,:,6] = torch.sum(torch.kron(self.layer0[:,:,3], self.layer0[:,:,6]), (-1, -2), True)
        self.layer0[:,:,7] = torch.sum(torch.kron(self.layer0[:,:,4], self.layer0[:,:,7]), (-1, -2), True)
        self.layer0[:,:,8] = torch.sum(torch.kron(self.layer0[:,:,5], self.layer0[:,:,8]), (-1, -2), True)
        self.layer0[:,:,9] = torch.sum(torch.kron(self.layer0[:,:,6], self.layer0[:,:,9]), (-1, -2), True)
        self.layer0[:,:,10] = torch.sum(torch.kron(self.layer0[:,:,7], self.layer0[:,:,10]), (-1, -2), True)
        self.layer0[:,:,11] = torch.sum(torch.kron(self.layer0[:,:,8], self.layer0[:,:,11]), (-1, -2), True)
        self.layer0[:,:,12] = torch.sum(torch.kron(self.layer0[:,:,9], self.layer0[:,:,12]), (-1, -2), True)
        self.layer0[:,:,13] = torch.sum(torch.kron(self.layer0[:,:,10], self.layer0[:,:,13]), (-1, -2), True)
        self.layer0[:,:,13] = torch.sum(torch.kron(self.layer0[:,:,11], self.layer0[:,:,13]), (-1, -2), True)
        self.layer0[:,:,13] = torch.sum(torch.kron(self.layer0[:,:,12], self.layer0[:,:,13]), (-1, -2), True)

        for i in range(13, self.depth - 1):
            self.layer0[:,:,(i + 1)] = torch.sum(torch.kron(self.layer0[:,:,i], self.layer0[:,:,(i + 1)]), (-1, -2), True)
        
        # check the predefined output neurons to see if they're ready to fire
        # if they are, then return the action(s) to take
        take_action = []
        
        #print(self.layer0)
        #print('layer0')
        for i in range(0, self.num_controls):
            if (self.layer0[self.controls[i][0], self.controls[i][1], self.controls[i][2]].item() > self.thresholds_pos[i, 0].item()):
                take_action.append(1)
                self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])] = self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])].item() - self.thresholds_pos[i,0]
            else:
                if (self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])].item() > self.thresholds_neg[i,0].item()):
                    take_action.append(0)
                else:
                    take_action.append(-1)
        
        # update the threshold and signal layers
        torch.mul(self.layer0, torch.mul(torch.mul(torch.atan(self.layer1), self.layer5), self.layer6), out=self.layer1)
        torch.mul(self.layer0, torch.mul(torch.mul(torch.atan(self.layer2), self.layer7), self.layer8), out=self.layer2)
        torch.mul(self.layer0, torch.mul(torch.mul(torch.atan(self.layer3), self.layer9), self.layer10), out=self.layer3)
        torch.mul(self.layer0, torch.mul(torch.mul(torch.atan(self.layer4), self.layer11), self.layer12), out=self.layer4)

        # update the emotion layers
        torch.mul(self.layer1, torch.mul(torch.mul(torch.atan(self.layer5), self.layer13), self.layer14), out=self.layer5)
        torch.mul(self.layer1, torch.mul(torch.mul(torch.atan(self.layer6), self.layer15), self.layer16), out=self.layer6)
        torch.mul(self.layer2, torch.mul(torch.mul(torch.atan(self.layer7), self.layer17), self.layer18), out=self.layer7)
        torch.mul(self.layer2, torch.mul(torch.mul(torch.atan(self.layer8), self.layer19), self.layer20), out=self.layer8)
        torch.mul(self.layer3, torch.mul(torch.mul(torch.atan(self.layer9), self.layer21), self.layer22), out=self.layer9)
        torch.mul(self.layer3, torch.mul(torch.mul(torch.atan(self.layer10), self.layer23), self.layer24), out=self.layer10)
        torch.mul(self.layer4, torch.mul(torch.mul(torch.atan(self.layer11), self.layer25), self.layer26), out=self.layer11)
        torch.mul(self.layer4, torch.mul(torch.mul(torch.atan(self.layer12), self.layer27), self.layer28), out=self.layer12)

        # update the personality layers
        torch.mul(self.layer5, torch.mul(torch.mul(torch.atan(self.layer13), self.layer29), self.layer30), out=self.layer13)
        torch.mul(self.layer5, torch.mul(torch.mul(torch.atan(self.layer14), self.layer31), self.layer32), out=self.layer14)
        torch.mul(self.layer6, torch.mul(torch.mul(torch.atan(self.layer15), self.layer33), self.layer34), out=self.layer15)
        torch.mul(self.layer6, torch.mul(torch.mul(torch.atan(self.layer16), self.layer35), self.layer36), out=self.layer16)
        torch.mul(self.layer7, torch.mul(torch.mul(torch.atan(self.layer17), self.layer37), self.layer38), out=self.layer17)
        torch.mul(self.layer7, torch.mul(torch.mul(torch.atan(self.layer18), self.layer39), self.layer40), out=self.layer18)
        torch.mul(self.layer8, torch.mul(torch.mul(torch.atan(self.layer19), self.layer41), self.layer42), out=self.layer19)
        torch.mul(self.layer8, torch.mul(torch.mul(torch.atan(self.layer20), self.layer43), self.layer44), out=self.layer20)
        torch.mul(self.layer9, torch.mul(torch.mul(torch.atan(self.layer21), self.layer45), self.layer46), out=self.layer21)
        torch.mul(self.layer9, torch.mul(torch.mul(torch.atan(self.layer22), self.layer47), self.layer48), out=self.layer22)
        torch.mul(self.layer10, torch.mul(torch.mul(torch.atan(self.layer23), self.layer49), self.layer50), out=self.layer23)
        torch.mul(self.layer10, torch.mul(torch.mul(torch.atan(self.layer24), self.layer51), self.layer52), out=self.layer24)
        torch.mul(self.layer11, torch.mul(torch.mul(torch.atan(self.layer25), self.layer53), self.layer54), out=self.layer25)
        torch.mul(self.layer11, torch.mul(torch.mul(torch.atan(self.layer26), self.layer55), self.layer56), out=self.layer26)
        torch.mul(self.layer12, torch.mul(torch.mul(torch.atan(self.layer27), self.layer57), self.layer58), out=self.layer27)
        torch.mul(self.layer12, torch.mul(torch.mul(torch.atan(self.layer28), self.layer59), self.layer60), out=self.layer28)

        
        return take_action
    
    
        
        
