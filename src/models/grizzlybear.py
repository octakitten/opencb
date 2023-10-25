import torch
import numpy as np

class grizzlybear():

    # dimensions of the neural space
    width = 255
    height = 255
    depth = 32
    # num_controls = 4

    threshhold = np.random.random()

    # controls output layer
    # 36 x 36
    output01 = (32,32,20)
    output01_thresh_positive = threshhold
    output01_thresh_negative = -threshhold
    print(output01_thresh_negative)
    # 36 x 96
    output02 = (32,96,20)
    output02_thresh_positive = threshhold
    output02_thresh_negative = -threshhold
    # 96 x 36
    output03 = (96,32,20)
    output03_thresh_positive = threshhold
    output03_thresh_negative = -threshhold
    # 96 x 96
    output04 = (96,96,20)
    output04_thresh_positive = threshhold
    output04_thresh_negative = -threshhold

    # neuron layer
    layer0 = torch.zeros((width, height, depth))
    print(layer0.shape)
    # threshold layers
    # positive thresh
    layer1 = torch.zeros((width, height, depth))
    # negative thresh
    layer2 = torch.zeros((width, height, depth))

    # emotion layers
    # positive thresh firing
    emotion1 = torch.zeros((width, height, depth))
    # positive thresh resting
    emotion2 = torch.zeros((width, height, depth))
    # negative thresh firing
    emotion3 = torch.zeros((width, height, depth))
    # negative thresh resting
    emotion4 = torch.zeros((width, height, depth))

    # personality layers

    # positive thresh firing is used
    personality1 = torch.zeros((width, height, depth))
    # positive thresh firing is unused
    personality2 = torch.zeros((width, height, depth))
    # positive thresh resting is used
    personality3 = torch.zeros((width, height, depth))
    # positive thresh resting is unused
    personality4 = torch.zeros((width, height, depth))
    # negative thresh firing is used
    personality5 = torch.zeros((width, height, depth))
    # negative thresh firing is unused
    personality6 = torch.zeros((width, height, depth))
    # negative thresh resting is used
    personality7 = torch.zeros((width, height, depth))
    # negative thresh resting is unused
    personality8 = torch.zeros((width, height, depth))

    # range of propensity to fire for personality layers
    pos_propensity = 20
    neg_propensity = -20

    def __init__(self):
            
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
        return
        
    def byop(self, personality1_1_1, personality1_1_2, personality1_2_1, 
                        personality1_2_2, personality2_1_1, personality2_1_2, 
                        personality2_2_1, personality2_2_2):
        # personality layers
        # positive thresh firing is used
        self.personality1 = personality1_1_1
        # positive thresh firing is unused
        self.personality2 = personality1_1_2
        # positive thresh resting is used
        self.personality3 = personality1_2_1
        # positive thresh resting is unused
        self.personality4 = personality1_2_2
        # negative thresh firing is used
        self.personality5 = personality2_1_1
        # negative thresh firing is unused
        self.personality6 = personality2_1_2
        # negative thresh resting is used
        self.personality7 = personality2_2_1
        # negative thresh resting is unused
        self.personality8 = personality2_2_2
        return
    
    def get_a_personality(self):
        return (self.personality1, self.personality2, self.personality3, self.personality4, self.personality5, self.personality6, self.personality7, self.personality8)


    def update(self, input_image):
        # add in the input image
        torch.add(self.layer0[:, :, 1],  input_image)

        #check which neurons are firing and which arent, do the stuff
        positive_firing = torch.greater(self.layer0, self.layer1)
        positive_resting = torch.less_equal(self.layer0, self.layer1)
        negative_firing = torch.less(self.layer0, self.layer2)
        negative_resting = torch.greater_equal(self.layer0, self.layer2)

        # keep track of the values of the firing neurons
        pos_fire_amt = torch.multiply(positive_firing, self.layer1)
        neg_fire_amt = torch.multiply(negative_firing, self.layer2)

        # apply the firing values to each of the near neighbors
        self.layer0 = torch.add(self.layer0, torch.roll(pos_fire_amt, 1, 0))
        self.layer0 = torch.add(self.layer0, torch.roll(neg_fire_amt, -1, 0))
        self.layer0 = torch.add(self.layer0, torch.roll(pos_fire_amt, 1, 1))
        self.layer0 = torch.add(self.layer0, torch.roll(neg_fire_amt, -1, 1))
        self.layer0 = torch.add(self.layer0, torch.roll(pos_fire_amt, 1, 2))
        self.layer0 = torch.add(self.layer0, torch.roll(neg_fire_amt, -1, 2))

        # update the threshold layers
        self.layer1 = torch.add(self.layer1, torch.multiply(positive_firing, self.emotion1))
        self.layer1 = torch.add(self.layer1, torch.multiply(positive_resting, self.emotion2))
        self.layer2 = torch.add(self.layer1, torch.multiply(negative_firing, self.emotion3))
        self.layer2 = torch.add(self.layer1, torch.multiply(negative_resting, self.emotion4))

        # figure out which emotions were used and which weren't
        # and then update them
        self.emotion1 = torch.add(torch.multiply(positive_firing, self.personality1), torch.multiply(positive_resting, self.personality3))
        self.emotion2 = torch.add(torch.multiply(positive_resting, self.personality2), torch.multiply(positive_firing, self.personality4))
        self.emotion3 = torch.add(torch.multiply(negative_firing, self.personality5), torch.multiply(negative_resting, self.personality7))
        self.emotion4 = torch.add(torch.multiply(negative_resting, self.personality6), torch.multiply(negative_firing, self.personality8))
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
        print(take_action)
        return take_action