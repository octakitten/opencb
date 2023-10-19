import numpy as np

class neural_space_16depth_1action_greyscale():

    # dimensions of the neural space
    width = 256
    height = 256
    depth = 16
    # num_controls = 4


    # controls output layer
    # 36 x 36
    output01 = (32,32,15)
    output01_thresh_positive = 180
    output01_thresh_negative = -180
    # 36 x 96
    output02 = (32,96,15)
    output02_thresh_positive = 180
    output02_thresh_negative = -180
    # 96 x 36
    output03 = (96,32,15)
    output03_thresh_positive = 180
    output03_thresh_negative = -180
    # 96 x 96
    output04 = (96,96,15)
    output04_thresh_positive = 180
    output04_thresh_negative = -180

    # neuron layer
    layer0 = np.zeros((width, height, depth))
    print(layer0.shape)
    # threshold layers
    # positive thresh
    layer1 = np.zeros((width, height, depth))
    # negative thresh
    layer2 = np.zeros((width, height, depth))

    # emotion layers
    # positive thresh firing
    layer1_1 = np.zeros((width, height, depth))
    # positive thresh resting
    layer1_2 = np.zeros((width, height, depth))
    # negative thresh firing
    layer2_1 = np.zeros((width, height, depth))
    # negative thresh resting
    layer2_2 = np.zeros((width, height, depth))

    # personality layers
    # positive thresh firing is used
    layer1_1_1 = np.zeros((width, height, depth))
    # positive thresh firing is unused
    layer1_1_2 = np.zeros((width, height, depth))
    # positive thresh resting is used
    layer1_2_1 = np.zeros((width, height, depth))
    # positive thresh resting is unused
    layer1_2_2 = np.zeros((width, height, depth))
    # negative thresh firing is used
    layer2_1_1 = np.zeros((width, height, depth))
    # negative thresh firing is unused
    layer2_1_2 = np.zeros((width, height, depth))
    # negative thresh resting is used
    layer2_2_1 = np.zeros((width, height, depth))
    # negative thresh resting is unused
    layer2_2_2 = np.zeros((width, height, depth))

    def __init__(self):
            
        # personality layers
        # positive thresh firing is used
        self.layer1_1_1 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # positive thresh firing is unused
        self.layer1_1_2 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # positive thresh resting is used
        self.layer1_2_1 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # positive thresh resting is unused
        self.layer1_2_2 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # negative thresh firing is used
        self.layer2_1_1 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # negative thresh firing is unused
        self.layer2_1_2 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # negative thresh resting is used
        self.layer2_2_1 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        # negative thresh resting is unused
        self.layer2_2_2 = np.random.uniform(low=-2, high=2, size=(self.width, self.height, self.depth))
        return
        
    def byop(self, personality1_1_1, personality1_1_2, personality1_2_1, 
                        personality1_2_2, personality2_1_1, personality2_1_2, 
                        personality2_2_1, personality2_2_2):
        # personality layers
        # positive thresh firing is used
        self.layer1_1_1 = personality1_1_1
        # positive thresh firing is unused
        self.layer1_1_2 = personality1_1_2
        # positive thresh resting is used
        self.layer1_2_1 = personality1_2_1
        # positive thresh resting is unused
        self.layer1_2_2 = personality1_2_2
        # negative thresh firing is used
        self.layer2_1_1 = personality2_1_1
        # negative thresh firing is unused
        self.layer2_1_2 = personality2_1_2
        # negative thresh resting is used
        self.layer2_2_1 = personality2_2_1
        # negative thresh resting is unused
        self.layer2_2_2 = personality2_2_2
        return


    def update(self, input_image):
        # add in the input image
        np.add(self.layer0[:, :, 1],  input_image)

        #check which neurons are firing and which arent, do the stuff
        positive_firing = np.greater(self.layer0, self.layer1)
        positive_resting = np.less_equal(self.layer0, self.layer1)
        negative_firing = np.less(self.layer0, self.layer2)
        negative_resting = np.greater_equal(self.layer0, self.layer2)

        # keep track of the values of the firing neurons
        pos_fire_amt = positive_firing * self.layer1
        neg_fire_amt = negative_firing * self.layer2

        # apply the firing values to each of the near neighbors
        self.layer0 += np.roll(pos_fire_amt, 1, axis=0)
        self.layer0 += np.roll(pos_fire_amt, -1, axis=0)
        self.layer0 += np.roll(pos_fire_amt, 1, axis=1)
        self.layer0 += np.roll(pos_fire_amt, -1, axis=1)
        self.layer0 += np.roll(pos_fire_amt, 1, axis=2)
        self.layer0 += np.roll(pos_fire_amt, -1, axis=2)

        # update the threshold layers
        self.layer1 += positive_firing * self.layer1_1
        self.layer1 += positive_resting * self.layer1_2
        self.layer2 += negative_firing * self.layer2_1
        self.layer2 += negative_resting * self.layer2_2

        # figure out which emotions were used and which weren't
        # and then update them
        self.layer1_1 += positive_firing * self.layer1_1_1 + positive_resting * self.layer1_2_1
        self.layer1_2 += positive_resting * self.layer1_1_2 + positive_firing * self.layer1_2_2
        self.layer2_1 += negative_firing * self.layer2_1_1 + negative_resting * self.layer2_2_1
        self.layer2_2 += negative_resting * self.layer2_1_2 + negative_firing * self.layer2_2_2

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
        return take_action