import numpy as np
import torch
from . import screen
from . import model as mdl

class forest():
    __DEBUG = True
    # the condition the blob has to meet to win the game
    victory_condition = False
    loss_condition = False
    # the instance of horse that'll play the game
    blob = 0
    victory_x = 0
    victory_y = 0
    starting_x = 0
    starting_y = 0
    min_dx = 0
    min_dy = 0

    width, height = 0, 0
    game_screen = 0
    
    # values that represent the brightness of the pixels where that object is
    tree = 0
    tile = 50
    player = 200
    flag = 255
    
    player_location_x = 0
    player_location_y = 0
    
    #get the current gamestate
    def screen(self):
        return self.game_screen
    
    
    #initializing the game requires giving it a model to play with
    def __init__(self, params, path=None):
        self.blob = mdl.ferret()
        if path==None:
            self.blob.create(params[0], params[1], params[2], params[3], params[4], params[5])
        else:
            self.blob.load(path)
        self.width = self.blob.width
        self.height = self.blob.height
        self.__create_starting_screen()
        self.__create_victory_condition()
        self.__choose_starting_location()
        screen.screen.save(self.game_screen, 'start_screen')
        return
    
    def __create_victory_condition(self):
        self.__choose_victory_location()
        self.game_screen[self.victory_x,self.victory_y] = self.flag
        screen.screen.save(self.game_screen, 'victory_condition')
        return
    
    def __loss_condition(self):
        if self.game_screen[self.player_location_x, self.player_location_y].item() == self.tree:
            return True
        else:
            return False
        
    def __win_condition(self):
        if (self.player_location_x == self.victory_x & self.player_location_y == self.victory_y):
            return True
        else:
            return False
    
    def __victory(self):
        if (self.__win_condition == True):
            return True
        else:
            return False
    
    def __screen_update(self, x, y):
        self.game_screen[self.player_location_x, self.player_location_y] = self.tile
        self.player_location_x += x
        self.player_location_y += y
        if (self.player_location_x < 0):
            self.player_location_x = 0
        if (self.player_location_x > self.width - 1):
            self.player_location_x = self.width - 1
        if (self.player_location_y < 0):
            self.player_location_y = 0
        if (self.player_location_y > self.height - 1):
            self.player_location_y = self.height - 1
        
        if (self.game_screen[self.player_location_x, self.player_location_y].item() == self.tree):
            self.loss_condition = True
        if (self.game_screen[self.player_location_x, self.player_location_y].item() == self.flag):
            self.victory_condition = True
        if (self.loss_condition == False & self.victory_condition == False):
            self.game_screen[self.player_location_x, self.player_location_y] = self.player
        return
    
    def __create_starting_screen(self):
        self.game_screen = torch.add(torch.zeros((self.width, self.height), device=torch.device('cuda'), dtype=torch.int16), self.tile)
        self.__create_forest()
        return
    
    def __check_valid_tree_spot(self, x, y):
        batch = []
        for i in range(-2, 2):
            for j in range(-2, 2):
                if ((x + i < self.width - 1) & (y + j < self.height - 1)):
                   if (self.game_screen[x + i, y + j].item() == self.tree):
                       return False
        return True
    
    def __check_close_to_tree(self, x, y):
        for i in range(-1, 1):
            for j in range(-1, 1):
               if ((x + i < self.width - 1) & (y + j < self.height - 1)):
                   if (self.game_screen[x + i, y + j].item() == self.tree):
                       return True
        return False
    
    def __create_forest(self):
        print('creating game state...')
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.__check_valid_tree_spot(i, j):
                    if np.random.randint(0, 100) < 50:
                        self.game_screen[i, j] = self.tree
                        #print('planting tree at ', i, j)

        return
        
    
    def __choose_starting_location(self):
        rand_x = np.random.randint(1, self.width)
        rand_y = np.random.randint(1, self.height)
        self.game_screen[rand_x, rand_y] = self.player
        self.player_location_x = rand_x
        self.player_location_y = rand_y
        return
    
    def __choose_victory_location(self):
        rand_x = np.random.randint(1, self.width)
        rand_y = np.random.randint(1, self.height)
        self.game_screen[rand_x, rand_y] = 255
        self.victory_x = rand_x
        self.victory_y = rand_y
        return
    

    def __blob_action(self, action):
        # decide on the action to take based on the input and 
        # convert that to an appropriate action for this game
        x, y = 0, 0
        if action[0] == 1:
            if self.__DEBUG: print("right 1")
            x += 1 
        if action[1] == 1:
            if self.__DEBUG: print("left 1")
            x += -1
        if action[2] == 1:
            if self.__DEBUG: print("up 1")
            y += 1
        if action[3] == 1:
            if self.__DEBUG: print("down 1")
            y += -1
        return x, y
    
    def __set_x(self, w):
        self.x = w
        return
    
    def __set_y(self, h):
        self.y = h
        return
    
    # if you want to change the model, do so with this function.
    def set_model(self, model):
        self.blob = model
        self.width = self.blob.width
        self.height = self.blob.height
        self.__create_starting_screen()
        self.__create_victory_condition()
        screen.screen.save(self.game_screen, 'start_screen')
        return
    
    def restart(self):
        self.loss_condition = False
        self.victory_condition = False
        self.__choose_starting_location()
        return
    
    # runs the game
    # if the model wins, it returns true
    # otherwise it returns false
    def play_game(self):

        # number of iterations the game will run for at max
        iter = 0
        max_iter = 1000

        # the previous action taken
        prev = torch.zeros(self.blob.num_controls).to(dtype=torch.float64, device=self.blob.device)
        # how many times the previous action has been the same as the current one
        combo = 0
        # how many times we'll allow it to combo the same action in a row
        max_combo = self.blob.width * 3
        # the previous x and y values
        prev_x = 0
        prev_y = 0
        
        min_dx = np.abs(self.victory_x - self.starting_x)
        min_dy = np.abs(self.victory_y - self.starting_y)
        
        # play the game until victory or until either combo gets too high or iterations finish
        while ((self.victory_condition == False) & (self.loss_condition == False)):
            act = self.blob.update(self.game_screen)
            if torch.equal(act, prev):
                combo += 1
            else:
                combo = 0
            
            if combo > max_combo:
                if self.__DEBUG: print("combo break")
                break

            prev = act
            
            x, y = self.__blob_action(act)
            self.__screen_update(x, y)
            
            x = np.abs(x - self.victory_x)
            y = np.abs(y - self.victory_y)
            if (x < prev_x):
                if self.__DEBUG: print("sense 0 positive sent")
                self.blob.sense(0, 255, True)
            if ( x > prev_x):
                if self.__DEBUG: print("sense 0 negative sent")
                self.blob.sense(0, 255, False)
            if (y < prev_y):
                if self.__DEBUG: print("sense 1 positive sent")
                self.blob.sense(1, 255, True)
            if (y > prev_y):
                if self.__DEBUG: print("sense 1 negative sent")
                self.blob.sense(1, 255, False)
                
            if (self.__check_close_to_tree(self.player_location_x, self.player_location_y) == True):
                if self.__DEBUG: print("sense 2 negative sent")
                self.blob.sense(2, 1000, False)
            
            if (x + y < self.min_dx + self.min_dy):
                self.min_dx = x
                self.min_dy = y
            self.prev_x = x
            self.prev_y = y
            iter += 1
            #print('movement turn - ', iter)
            if iter > max_iter:
                break
            if (self.loss_condition == True):
                break
            if (self.victory_condition == True):
                break
        
        # the blob satisfies the victory condition, which is to put the white dot at the location of the victory condition
        # then we win, and we print out the personality layers

        # otherwise we lose, and just do nothing with it.
        if (self.victory_condition == True):
            if self.__DEBUG: print("victory! it took this many turns:")
            if self.__DEBUG: print(iter)
            return True
        else:
            if self.__DEBUG: print("defeat! after this many turns:")
            if self.__DEBUG: print(iter)
            #screen.save(self.game_screen, 'end_screen')
        return False
