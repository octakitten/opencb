import torch
import numpy as np
import os

class ferret():
        
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

    layers = []
    firings = []

    # range of propensity to fire for personality layers
    pos_propensity = 0
    neg_propensity = 0
    
    propensity = 0
    
    outputs = 0

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
        '''
        Save a model to a file.

        :Parameters:
        path (str): the path to the directory to save the model files

        :Returns:
        none
        '''
        if (os.path.exists(path) == False):
            os.makedirs(path)
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
        for i in range(0, len(self.layers)):
            torch.save(self.layers[i], path + '/layer' + str(i) + '.pth')
        return
    
    def load(self, path):
        '''
        Load a model from a file.

        :Parameters:
        path (str): the path to the directory containing the model files

        :Returns:
        none
        '''
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
        for i in range(0, len(self.layers)):
            self.layers.append(torch.load(path + '/layer' + str(i) + '.pth'))
        
        for i in range(0, 8):
            self.firing = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)

        return
        
    def clear(self):
        for i in range(0, 28):
            self.layers.append(torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device))
        return
        
    def copy(self, model):
        '''
        Copy a model's parameters to a new model.
        
        :Parameters:
        model (ferret): the model to copy
        
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
        self.layers = []
        for i in range(0, len(model.layers)):
            self.layers.append(model.layers[i])
        for i in range(0, 8):
            self.firing = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        
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
     
    def __new_controls(self):
        self.controls = []
        for i in range(0, self.num_controls):
            wegood = False
            newctl = 0
            while wegood == False:
                newctl = (np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), int((self.depth - 1) * .37))
                wegood = True
                for ctl in self.controls:
                    if ctl == newctl:
                        wegood = False
            self.controls.append(newctl)
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
        from layers[0] to emotion8 to personality8, but the personality layers we initialize to random values. These random values
        should range from 1 to n for the positive personality layers, and 1 to 1/n for the negative personality layers. in order to 
        achieve this, we first generate random values between 0 and 1, then for the positive layers we multiply by n and add 1, and for
        the negative layers we divide by n and subtract from 1. This will give us the desired range of values for the personality layers.
        '''
        for i in range(0, 60):
            self.layers.append(torch.tensor(data=1, device=self.device))
        for i in range(0, 60):
            self.layers[i] = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
        for i in range(29, 60):
            self.layer0 = torch.tensor(data=1, device=self.device)
            random_gen = torch.Generator(device=self.device)
            random_gen.seed()
            self.layers[i] = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float64, device=self.device))).to(dtype=torch.int16)
        
        for i in range(0, 8):
            self.firing = torch.zeros((self.width, self.height, self.depth), dtype=torch.int16, device=self.device)
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
        return

    def __new_sensations(self):
        self.sensations = []
        for i in range(0, self.num_sensations):
            self.sensations.append((np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), int(self.depth * .5)))
        return

    def __pos_sensation(self, sense_num, amt):
        torch.add(self.layer0[self.sensations[sense_num]], amt, out=self.layers[0][self.sensations[sense_num]])
        return

    def __neg_sensation(self, sense_num, amt):
        torch.subtract(self.layer0[self.sensations[sense_num]], amt, out=self.layers[0][self.sensations[sense_num]])
        return
    
    def sense(self, sense_num, amt, pos):    
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
        degree = 1 / fraction
        model = ferret()
        model.copy(self)
        model.__new_thresholds()
        model.__new_propensity()
        torch.divide(model.thresholds_pos, fraction, out=model.thresholds_pos)
        torch.divide(model.thresholds_neg, fraction, out=model.thresholds_neg)
        torch.divide(self.thresholds_pos, degree + 1, out=self.thresholds_pos)
        torch.divide(self.thresholds_neg, degree + 1, out=self.thresholds_neg)
        for i in range(29, 60):
            temp = torch.tensor(data=1, device=self.device)
            random_gen = torch.Generator(device=self.device)
            random_gen.seed()
            temp = torch.multiply(other=self.pos_propensity[0,0], input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float64, device=self.device))).to(dtype=torch.int16)
            temp = torch.divide(temp, fraction,).to(dtype=torch.int16)
            self.layers[i] = torch.add(self.layers[i], model.layers[i], out=self.layers[i])
        return

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

        self.outputs = torch.zeros(self.num_controls).to(dtype=torch.int16, device=self.device)

        if (torch.is_tensor(input_image) == False):
            return -1
        # add in the input image
        input_image.to(dtype=torch.int16, device=self.device)
        input_tensor = torch.tensor(data=1, device=self.device)
        
        input_tensor = torch.clone(input_image, ).detach().to(dtype=torch.int16, device=self.device)
        try:
            torch.add(self.layers[0][:, :, 0],  input_tensor[0,:,:], out=self.layers[0][:, :, 0])
            torch.add(self.layers[0][:, :, 1],  input_tensor[1,:,:], out=self.layers[0][:, :, 1])
            torch.add(self.layers[0][:, :, 2],  input_tensor[2,:,:], out=self.layers[0][:, :, 2])
        except:
            torch.add(self.layers[0][:, :, 0],  input_tensor, out=self.layers[0][:, :, 0])

        #check which neurons are firing and which arent, do the stuff
        torch.greater(self.layers[0], self.layers[1], out=self.firing[0])
        torch.less_equal(self.layers[0], self.layers[1], out=self.firing[1])
        torch.less(self.layers[0], self.layers[2], out=self.firing[2])
        torch.greater_equal(self.layers[0], self.layers[2], out=self.firing[3])

        # keep track of the threshold values of the firing neurons
        torch.multiply(self.firing[0], self.layers[1], out=self.firing[4])
        torch.multiply(self.firing[2], self.layers[2], out=self.firing[5])
        
        #self.pos_fre_amt = torch.div(self.pos_fire_amt, 6).to(dtype=torch.int16)
        #self.neg_fire_amt = torch.div(self.neg_fire_amt, 6).to(dtype=torch.int16)

        # use the firing multipliers to change the output values of the firing neurons
        torch.add(self.firing[0], self.layers[3], out=self.firing[7])
        torch.sub(self.firing[2], self.layers[4], out=self.firing[8])

        # apply the firing values to each of the near neighbors
        temp = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.int16)
        for i in range(0, 4):
            torch.add(self.layers[0], torch.roll(self.firing[7], (-1 ** i), int(i/2), out=temp))
            torch.sub(self.layers[0], torch.roll(self.firing[8], (-1 ** i), int(i/2), out=temp))
        
        # check the predefined output neurons to see if they're ready to fire
        # if they are, then return the action(s) to take
        for i in range(0, self.num_controls):
            if (self.layer0[self.controls[i][0], self.controls[i][1], self.controls[i][2]].item() > self.thresholds_pos[i, 0].item()):
                self.outputs[i] = 1
                self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])] = self.layer0[(self.controls[i][0], self.controls[i][1], self.controls[i][2])].item() - self.thresholds_pos[i,0]
            else:
                if (self.layers[0][(self.controls[i][0], self.controls[i][1], self.controls[i][2])].item() > self.thresholds_neg[i,0].item()):
                    self.outputs[i] = 0
                else:
                    self.outputs[i] = -1
        
        # update layers[0] by decrementing all the firing neurons by their firing amount
        torch.sub(self.layer[0], self.firing[5], out=self.layer[0])
        torch.sub(self.layer[0], self.firing[6], out=self.layer[0])

        # update the threshold layers
        for i in range(0, 8):
            torch.add(torch.mul(self.firing[int(i/2)], self.layers[5 + i]), self.layers[int(i/2)], out=self.layers[int(i/2)])
        
        # figure out which emotions were used and which weren't
        # and then update them according to the personality values
        
        for i in range(0, 16):
            torch.add(torch.mul(self.firing[i % 4], self.layers[13 + i]), self.layers[int(i/4) + 5], out=self.layers[int(i/4) + 5])
        
        # now update the personality values according to their associated dna values
        
        for i in range(0, 32):
            torch.add(torch.mul(self.firing[i % 4], self.layers[29 + i]), self.layers[int(i/2) + 13], out=self.layers[int(i/2) + 13])

        return self.outputs

    def backprop(self, guess, answer, constant = None):
        '''
        Backpropagation function

        :Parameters:
        answer (tensor): the correct answer to the model's output

        :Returns:
        none

        :Comments:
        This does not work like traditional backprop does. We have to sort of approximate the process of 
        taking a gradient because this system is not differentiable over time. In fact, it's not even 
        representable by a function in the first place. Only by its own algorith I believe. In order to
        achieve the results we would get from backprop in a more usual neural network design, I think
        we have to represent the gradient in a probability graph rather than a vector graph. Essentially,
        we treat the actual results of the network as unknowably random, and predict the probability 
        of a specific neuron firing based on the probability that the 6 neurons its in contact with
        would have fired. We look at what would have needed to happen for each neuron to fire, and assign a 
        probability to each of those things. Then we look at what actually happened and adjust the dna
        values to try to move those probabilities closer to the results we need, rather than the results
        we got.

        The nice thing here is that we can use this backprop function immediately following a call to the
        update function. This means all the data we need should be still in place and unchanged. First up
        we identify which neuron(s) needed to fire, and which did not. Then we try to look at dna value
        configurations that would have resulted in the correct firing state last turn. We adjust the dna
        values of its 6 in-contact neurons to a state that would have resulted in the neuron firing... but
        well, here's where things get complicated. The probability that these neurons fire depends on their
        contact neurons as well. It starts to become prohibitively difficult to pick out a specific 
        configuration that we want to shoot for when iterating over the whole network. Thus we need to nudge
        neurons rather than set theto a specific value. If a neuron needed to fire and didn't, we can 
        change the dna values of its contact neurons and then propagate out from them. This means that for each 
        neuron that we need to change, we also need to change its own contact neurons. This will have a 
        recursive effect on the origin neuron too, which is fine I think. 

        Anyway, to lay out how this process will work: pick an output neuron and an answer that it should have
        given but didn't. Nudge its contact neurons in a direction that would have increased the probability
        of the output neuron firing. Then, treat each of those contact neurons as an output neuron that 
        should have had a different answer, depending no what we needed from them. Nudge their contact neurons
        appropriately too, then continue this process throughout the whole network.

        How much a given neuron needs to be nudged should depend on its contribution to the probability 
        of the output neuron firing. This will probably best be represented with a function that 
        depends on the distance between the neurons actual value and the firing value it needed to get to.
        If that value is closer, then the neuron was more likely to fire, and so was contributing more and
        should be nudged more. Note that the contribution of the output neuron itself is the highest 
        contribution of all, and so it should be nudged the most. 

        Now, how can we implement this algorithm? We need to start from the output neuron and work outward
        through the network. We'll need to iterate in steps that sort of add 3d layers on top of the part 
        of the network we just nudged. The formual we want to use should look like: 
        Nudge ~= k / ((On - Oa) * (Cn - Ca))
        where k is some constant that we can manipulate, O is the output neuron and C is the contact 
        neuron, and n subscript is the needed val while a subscript is the actual val.

        One thing to note is that there are 4 states a neuron can potentially be in, and it will be in
        2 of them at a time at all times. This means that there's two states we need to nudge towards:
        the correct firing state and the correct resting state. These will be split between the positive
        and negative states though, so we nudge to the right positive state and then the right negative state
        seperately.

        Ok, let's get to it I guess.
        '''

        # first, we pick an item from the output layer and compare it to its corresponding item in the answer key.
        # if they match, cool, on to the next one. if they dont, nudge the dna values in an appropriate
        # direction.
        cons = 0.37
        if constant != None: cons = constant

        for i in range(0, len(self.outputs)):
            if (guess[i] != answer[i]):
                diff = int(abs(guess[i] - answer[i]))
                for j in range(0, diff):
                    if guess[i] == 0: cons = cons / 2
                    # if the output was 1 and it should have been 0, then we need to nudge the dna values
                    # of the contact neurons in the direction that would have made the output neuron not fire
                    # if (self.outputs[i] == 1):
                    # first, we need to find the contact neurons and nudge them in the right direction
                    # to make the output neuron not fire
                    '''
                    cons = 1
                    outx = self.controls[i][0]
                    outy = self.controls[i][1]
                    outz = self.controls[i][2]
                    dx = 1
                    dy = 0
                    dz = 0
                    # oloss should be the difference between the output neurons actual value and the closest value it could have had that would have satisfied the conditions we want, which in this case is for it to not fire.
                    # thus it should be the remaining value left after the neuron fired
                    oloss = self.layers[0][outx, outy, outz]
                    
                    # closs is a bit more complicated, its the values of the contact neuron and its dna that would have helped move the output neuron to the state it should have been in
                    # since the output fired when it shouldnt have, the closs should be something that would help minimize the oloss
                    # thus, when the output neuron shouldnt fire, the contact neurons shouldnt fire either... but only if their firing would have helped the output neuron not fire
                    # we will have to take the firing value of the contact neuron and 
                    closs1 = self.layers[0][(outx + dx), (outy + dy), (outz + dz)] * (self.positive_firing[(outx + dx), (outy + dy), (outz + dz)] 
                    closs2 = self.layers[0][(outx + dx), (outy + dy), (outz + dz)] * (self.posiitive_resting[(outx + dx), (outy + dy), (outa + dz)])
                    closs3 = self.layers[0][(outx + dx), (outy + dy), (outz + dz)] * (self.negative_firing[(outx + dx), (outy + dy), (outz + dz)])
                    closs4 = self.layers[0][(outx + dx), (outy + dy), (outz + dz)] * (self.negative_resting[(outx + dx), (outy + dy), (outz + dz)])

                    nudge1 = cons / (oloss * closs1)
                    nudge2 = cons / (oloss * closs2)
                    nudge3 = cons / (oloss * closs3)
                    nudge4 = cons / (oloss * closs4)

                    self.dna1[(outx + dx), (outy + dy), (outz + dz)] += nudge1
                    self.dna2[(outx + dx), (outy + dy), (outz + dz)] += nudge2
                    self.dna3[(outx + dx), (outy + dy), (outz + dz)] += nudge3
                    self.dna4[(outx + dx), (outy + dy), (outz + dz)] += nudge4
                    '''
                    # after this we just repeat this process across the entire network.
                    # thing is, this whole process is incredibly slow in native python
                    # we need to run this using tensors instead.
                    # it should look like this:
                    for i in range(0, 32):
                        self.layers[29+i] = torch.add(self.layers[29+i], torch.mul(cons, torch.sqrt(torch.mul(self.layers[0], self.firing[i % 4]))), ).to(dtype=torch.int16)

                    # and there we go! the only thing left to note is the inclusion of a scaling factor "cons" in the equations. you should be
                    # able to set cons to a value between 0 and 1 to slow down the backprop process's effect per use of the function.
                    # it wont work well if you set it above 1 since that could have unintended effects, and setting it to a negative value
                    # will just have the effect of driving the model's training further from the desired results instead. 
            return




class hamster():
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

    propensity = 100
    
    layers = []

    outputs = 0

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
                newctl = (np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), int(self.depth - 1))
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
        self.pos_propensity = self.pos_propensity[0][0]
        self.neg_propensity = self.neg_propensity[0][0]
        return
    
    def __new_dna(self):
        for i in range(0, 61):
            self.layers.append(torch.tensor(data=1, device=self.device))
            self.layers[i] = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float64, device=self.device)
        for i in range(29, 61):
            random_gen = torch.Generator(device=self.device)
            random_gen.seed()
            self.layers[i] = torch.multiply(other=self.pos_propensity, input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float64, device=self.device)))

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
        for i in range(0, 61):
            torch.save(self.layers[i], path + '/layer' + str(i) + '.pth')
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
        self.control_thresholds_pos = torch.load(path + '/control_thresholds_pos.pth')
        self.control_thresholds_neg = torch.load(path + '/control_thresholds_neg.pth')
        for i in range(0, 61):
            self.layers.append(torch.load(path + '/layer' + str(i) + '.pth'))
        return
        
    def __new_thresholds(self):
        random_gen = torch.Generator(device=self.device)
        random_gen.seed()
        self.control_thresholds_pos = torch.tensor(data=1, device=self.device)
        self.control_thresholds_neg = torch.tensor(data=1, device=self.device)
        self.control_thresholds_pos = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=self.device)
        torch.add(torch.mul(self.control_thresholds_pos, self.pos_propensity, out=self.control_thresholds_pos), 1, out=self.control_thresholds_pos)
        random_gen.seed()
        self.control_thresholds_neg = torch.rand(size=(self.num_controls, self.num_controls), generator=random_gen, device=self.device)
        torch.subtract(-1, torch.divide(self.control_thresholds_neg, self.neg_propensity, out=self.control_thresholds_neg), out=self.control_thresholds_neg)
        return

    def __new_sensations(self):
        self.sensations = []
        for i in range(0, self.num_sensations):
            self.sensations.append((np.random.randint(low=1, high=self.width), np.random.randint(low=1, high=self.height), np.random.randint(low=1, high=self.depth)))
        return

    def __pos_sensation(self, sense_num, amt):
        torch.add(self.layers[0][self.sensations[sense_num]], amt, out=self.layers[0][self.sensations[sense_num]])
        return

    def __neg_sensation(self, sense_num, amt):
        torch.subtract(self.layers[0][self.sensations[sense_num]], amt, out=self.layers[0][self.sensations[sense_num]])
        return

    def copy(self, model):
        '''
        Copy a model's parameters to a new model.
        
        :Parameters:
        model (ferret): the model to copy
        
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
        for i in range(0, 61):
            self.layers[i] = model.layers[i]
        
        return

    def clear(self):
        for i in range(0, 28):
            self.layers[i] = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float64, device=self.device)
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

        threshp = self.control_thresholds_pos
        threshn = self.control_thresholds_neg
        self.__new_thresholds()
        torch.divide(self.control_thresholds_pos, fraction, out=self.control_thresholds_pos)
        torch.divide(self.control_thresholds_neg, fraction, out=self.control_thresholds_neg)
        torch.add(self.control_thresholds_pos, threshp, out=self.control_thresholds_pos)
        torch.add(self.control_thresholds_neg, threshn, out=self.control_thresholds_neg)

        for i in range(29, 61):
            temp = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float64, device=self.device)
            random_gen = torch.Generator(device=self.device)
            random_gen.seed()
            torch.multiply(input=torch.sub(other=0.5, input=torch.rand(size=(self.width, self.height, self.depth), generator=random_gen, dtype=torch.float64, device=self.device)), other=self.pos_propensity, out=temp)
            torch.divide(temp, fraction, out=temp)
            torch.add(self.layers[i], temp, out=self.layers[i])
        for i in range(0, 28):
            self.layers[i] = torch.zeros(size=(self.width, self.height, self.depth), dtype=torch.float64, device=self.device)
        return
        
    def update(self, input_image):
        if (torch.is_tensor(input_image) == False):
            return -1
        # add in the input image
        input_image.to(dtype=torch.float64, device=self.device)
        input_tensor = torch.tensor(input_image, device=self.device)
        #print(input_image.device)
        #print(input_tensor)
        #torch.div(input_tensor, 255, out=input_tensor)
        #torch.mul(input_tensor, self.bounds, out=input_tensor)
        #torch.add(input_tensor, torch.ones(size=input_image.size(), device=self.device), out=input_tensor)
        #print(input_tensor)
        #print('layers[0]')
        #print(self.layers[0])o
        try:
            torch.add(self.layers[0][:, :, 0],  input_tensor[0, :, :], out=self.layers[0][:, :, 0])
            torch.add(self.layers[0][:, :, 1],  input_tensor[1, :, :], out=self.layers[0][:, :, 1])
            torch.add(self.layers[0][:, :, 2],  input_tensor[2, :, :], out=self.layers[0][:, :, 2])
        except:
            torch.add(self.layers[0][:, :, 0],  input_tensor[0, :, :], out=self.layers[0][:, :, 0])

        print('input tensor')
        print(input_tensor)

        # update layers[0] based on the arctan function we're using, as well as inputs from the threshold and signal layers
        torch.add(torch.add(torch.atan(torch.add(self.layers[0], self.layers[1])), self.layers[3]), torch.add(torch.atan(torch.add(self.layers[0], self.layers[2])), self.layers[4]), out=self.layers[0])

        # do some rolls to simulate neurons sending messages to each other
        '''
        temp = torch.zeros(size=(self.width, self.height, self.depth), device=self.device, dtype=torch.float64)
        torch.add(self.layers[0], torch.roll(self.layers[0], 1, 0), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], -1, 0), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], 1, 1), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], -1, 1), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], 1, 2), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], -1, 2), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], 1, 0), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], -1, 0), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], 1, 1), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], -1, 1), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], 1, 2), out=temp)
        torch.add(self.layers[0], torch.roll(self.layers[0], -1, 2), out=temp)
        torch.add(self.layers[0], temp, out=self.layers[0])
        '''
        # so we're going to try using kron sums again... i want to propagate information forward from the input images through the whole network
        # in one go... i think this way we can get results faster per turn than having to wait for the model to slowly propagate information through
        # layer0...
        # its going to be more resource intensive but i think in the long run it wont actually slow the overall process down.
        for i in range(0, 3):
            torch.add(torch.sum(torch.kron(self.layers[0][:, :, i], self.layers[0][:, :, (i+1)])), self.layers[0][:, :, (i+1)], out=self.layers[0][:, :, (i+1)])

        for i in range(3, int((self.depth - 1)/2)):
            i = i * 2
            torch.add(torch.sum(torch.kron(self.layers[0][:, :, i], self.layers[0][:, :, (i+1)])), self.layers[0][:, :, (i+1)], out=self.layers[0][:, :, (i+1)])
            torch.add(self.layers[0][:,:,i+1], self.layers[0][:,:,(i+2)], out=self.layers[0][:,:,i+2])

        
        self.outputs = torch.zeros(self.num_controls).to(dtype=torch.float64, device=self.device)
        for i in range(0, self.num_controls):
            self.outputs[i] = self.layers[0][self.controls[i][0], self.controls[i][1], self.controls[i][2]].item()
        softmax = torch.nn.Softmax(dim=0)
        self.outputs = softmax(self.outputs)
        
        # update the threshold and signal layers
        for i in range (1, 5):
            torch.add(torch.atan(torch.add(self.layers[i], torch.add(self.layers[0], self.layers[5 + 2*(i-1)]))), torch.add(self.layers[i], self.layers[6 + 2*(i-1)]), out=self.layers[0])

        # update the emotion layers
        for i in range (5, 13):
            torch.add(torch.atan(torch.add(self.layers[i], torch.add(self.layers[int((i - 3)/2)], self.layers[(12 + (i - 5)*2)]))), torch.add(self.layers[int((i - 3)/2)], self.layers[(13 + (i - 5)*2)]), out=self.layers[i])

        # update the personality layers
        for i in range(13, 29):
            torch.add(torch.atan(torch.add(self.layers[i], torch.add(self.layers[int((i - 1) / 2)], self.layers[(29 + (i - 13)*2)]))), torch.add(self.layers[int((i - 1) / 2)], self.layers[(30 + (i - 13)*2)]), out=self.layers[i])

        print('layers[0]')
        print(self.layers[0])
        return self.outputs

    def backprop(self, guess, answer, constant=None):
        '''
        Backpropagate the model to adjust its dna values based on the answer it should have given.

        :Parameters:
        guess (list): the output of the model
        answer (list): the answer the model should have given
        constant (float): the constant to adjust the backpropagation by

        :Returns:
        none

        :Comments: 
        This function will be used to adjust the model's dna values based on the answer it should have given.
        The model will adjust its dna values based on the difference between the answer it should have given
        and the answer it actually gave. This will be done by nudging the dna values of the neurons that
        contributed to the model's output. The amount that the dna values will be nud
        '''

        cons = 0.37
        if constant != None: cons = constant

        for i in range(0, len(self.outputs)):
            if (guess[i] != answer[i]):
                diff = abs(guess[i] - answer[i])
                print('diff')
                print(diff)
                if guess[i] == 0: cons = cons / 2

                # self.dna1 = torch.add(self.dna1, torch.div(cons, torch.mul(self.layers[0], torch.mul(self.layers[0], self.positive_firing))), ).to(dtype=torch.int16)
                # heres the actual homework we need to copy. this needs to be adapted for our puroses here. it wont work as is though, we dont have 
                # records of which neurons fired or didnt in this model, since practically speaking they all fired to some degree. 

                # one of the accomplishments of back propagation is that it very efficiently targets the parts of the network that contribute to 
                # the error in the output. we will need to find a way to do something similar here.
                # i think to start with we should find the absolute values of the neurons in layers[0]. we can decrement them the most and leave others alone, at least relatively so

                # i think we want the decrement to be a function of value @ layers[0] ^ 2, so that it weights the decrement more towards the neurons that contributed the most,
                # it should also be proportional to the abs size of the dna values, so that the dna values which contributed the most to the output are the ones that are most affected.
                # and of course we include a constant cons to scale the intensity of the backpropagatiuon

                # i suppose if i really want to take a gradient, i can do that if i change up how this model propagates value through layer0.
                # if i use kron sums it might work...
                # 
                # yea... lets see if i can change up how this works a little bit...
                # alright so ive got kron sums going now. now we can actually attribute the values in each of the outputs to a continuous function of the inputs. i should also remember
                # to put the outputs on the final layer while im remembering it.

                # nudge = cons / layer0 ^ 2 * dna

                for k in range(29, 61):
                    torch.mul(self.layers[k], torch.mul(torch.div(cons, torch.mul(torch.pow(torch.abs(self.layers[0]), .5), torch.pow(torch.abs(self.layers[k]), .5))), (1 - diff)), out=self.layers[k])
        return
