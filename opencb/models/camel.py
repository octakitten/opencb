import numpy as np

class camel():

    # dimensions of the neural space
    width = 255
    height = 255
    depth = 32
    # num_controls = 4


    # controls output layer
    # 36 x 36
    output01 = (32,32,20)
    threshold = np.random.uniform(low=-255, high=255)
    output01_thresh_positive = threshold
    output01_thresh_negative = -threshold
    # 36 x 96
    output02 = (32,96,20)
    output02_thresh_positive = threshold
    output02_thresh_negative = -threshold
    # 96 x 36
    output03 = (96,32,20)
    output03_thresh_positive = threshold
    output03_thresh_negative = -threshold
    # 96 x 96
    output04 = (96,96,20)
    output04_thresh_positive = threshold
    output04_thresh_negative = -threshold

    # neuron layer
    layer0 = np.zeros((width, height, depth))
    #print(layer0.shape)
    # threshold layers
    # positive thresh
    layer1 = np.zeros((width, height, depth))
    # negative thresh
    layer2 = np.zeros((width, height, depth))

    # emotion layers
    # positive thresh firing
    emotion1 = np.zeros((width, height, depth))
    # positive thresh resting
    emotion2 = np.zeros((width, height, depth))
    # negative thresh firing
    emotion3 = np.zeros((width, height, depth))
    # negative thresh resting
    emotion4 = np.zeros((width, height, depth))

    # personality layers

    # positive thresh firing is used
    personality1 = np.zeros((width, height, depth))
    # positive thresh firing is unused
    personality2 = np.zeros((width, height, depth))
    # positive thresh resting is used
    personality3 = np.zeros((width, height, depth))
    # positive thresh resting is unused
    personality4 = np.zeros((width, height, depth))
    # negative thresh firing is used
    personality5 = np.zeros((width, height, depth))
    # negative thresh firing is unused
    personality6 = np.zeros((width, height, depth))
    # negative thresh resting is used
    personality7 = np.zeros((width, height, depth))
    # negative thresh resting is unused
    personality8 = np.zeros((width, height, depth))

    # range of propensity to fire for personality layers
    propensity = np.random.uniform(low=1, high=255)
    #print('propensity')
    #print(propensity)

    def __init__(self):
            
        # personality layers
        # positive thresh firing is used
        self.personality1 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # positive thresh firing is unused
        self.personality2 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # positive thresh resting is used
        self.personality3 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # positive thresh resting is unused
        self.personality4 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # negative thresh firing is used
        self.personality5 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # negative thresh firing is unused
        self.personality6 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # negative thresh resting is used
        self.personality7 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        # negative thresh resting is unused
        self.personality8 = np.random.uniform(low=-self.propensity, high=self.propensity, size=(self.width, self.height, self.depth))
        return
        
    def byop(self, personality_tuple):
        # personality layers
        # positive thresh firing is used
        self.personality1 = personality_tuple[0]
        # positive thresh firing is unused
        self.personality2 = personality_tuple[1]
        # positive thresh resting is used
        self.personality3 = personality_tuple[2]
        # positive thresh resting is unused
        self.personality4 = personality_tuple[3]
        # negative thresh firing is used
        self.personality5 = personality_tuple[4]
        # negative thresh firing is unused
        self.personality6 = personality_tuple[5]
        # negative thresh resting is used
        self.personality7 = personality_tuple[6]
        # negative thresh resting is unused
        self.personality8 = personality_tuple[7]
        return


    def update(self, input_image):
        # add in the input image
        if (isinstance(input_image, np.ndarray) == False):
            return -1
        np.add(self.layer0[:, :, 1],  input_image)

        #check which neurons are firing and which arent, do the stuff
        positive_firing = np.greater(self.layer0, self.layer1)
        positive_resting = np.less_equal(self.layer0, self.layer1)
        negative_firing = np.less(self.layer0, self.layer2)
        negative_resting = np.greater_equal(self.layer0, self.layer2)

        # keep track of the values of the firing neurons
        pos_fire_amt = np.multiply(positive_firing, self.layer1)
        neg_fire_amt = np.multiply(negative_firing, self.layer2)

        # apply the firing values to each of the near neighbors
        self.layer0 = np.add(self.layer0, np.roll(pos_fire_amt, 1, axis=0))
        self.layer0 = np.add(self.layer0, np.roll(neg_fire_amt, -1, axis=0))
        self.layer0 = np.add(self.layer0, np.roll(pos_fire_amt, 1, axis=1))
        self.layer0 = np.add(self.layer0, np.roll(neg_fire_amt, -1, axis=1))
        self.layer0 = np.add(self.layer0, np.roll(pos_fire_amt, 1, axis=2))
        self.layer0 = np.add(self.layer0, np.roll(neg_fire_amt, -1, axis=2))

        # update the threshold layers
        self.layer1 = np.add(self.layer1, np.multiply(positive_firing, self.emotion1))
        self.layer1 = np.add(self.layer1, np.multiply(positive_resting, self.emotion2))
        self.layer2 = np.add(self.layer1, np.multiply(negative_firing, self.emotion3))
        self.layer2 = np.add(self.layer1, np.multiply(negative_resting, self.emotion4))

        # figure out which emotions were used and which weren't
        # and then update them
        self.emotion1 = np.add(np.multiply(positive_firing, self.personality1), np.multiply(positive_resting, self.personality3))
        self.emotion2 = np.add(np.multiply(positive_resting, self.personality2), np.multiply(positive_firing, self.personality4))
        self.emotion3 = np.add(np.multiply(negative_firing, self.personality5), np.multiply(negative_resting, self.personality7))
        self.emotion4 = np.add(np.multiply(negative_resting, self.personality6), np.multiply(negative_firing, self.personality8))
        '''
        self.emotion1 += positive_firing * self.personality1 + positive_resting * self.personality3
        self.emotion2 += positive_resting * self.personality2 + positive_firing * self.personality4
        self.emotion3 += negative_firing * self.personality5 + negative_resting * self.personality7
        self.emotion4 += negative_resting * self.personality6 + negative_firing * self.personality8
        '''
        # check the predefined output neurons to see if they're ready to fire
        # if they are, then return the action(s) to take
        # the primes are just an easy way to return multiple values as an integer,
        # don't worry about it

        take_action = 1
        # first neuron activated, take action 1
        if (self.layer0[self.output01] > self.output01_thresh_positive):
            take_action = take_action * 2
        # first neuron activated negatively, take action 1 in the opposite direction
        if (self.layer0[self.output01] < self.output01_thresh_negative):
            take_action = take_action * 3
        # action 2
        if (self.layer0[self.output02] > self.output02_thresh_positive):
            take_action = take_action * 5
        # negative of action 2
        if (self.layer0[self.output02] < self.output02_thresh_negative):
            take_action = take_action * 7
        # action 3
        if (self.layer0[self.output03] > self.output03_thresh_positive):
            take_action = take_action * 11
        # negative of action 3
        if (self.layer0[self.output03] < self.output03_thresh_negative):
            take_action = take_action * 13
        # action 4
        if (self.layer0[self.output04] > self.output04_thresh_positive):
            take_action = take_action * 17
        # negative of action 4
        if (self.layer0[self.output04] < self.output04_thresh_negative):
            take_action = take_action * 19
        #print(take_action)
        return take_action