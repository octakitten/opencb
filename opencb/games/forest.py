import numpy as np
import torch
from ..utilities import screen
from ..models import general_dev

class forest():
    # the condition the blob has to meet to win the game
    victory_condition = 0
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
    player = 250
    flag = 255
    
    player_location_x = 0
    player_location_y = 0
    
    #get the current gamestate
    def screen(self):
        return self.game_screen
    
    
    #initializing the game requires giving it a model to play with
    def __init__(self, model):
        self.blob = model
        self.width = self.blob.width
        self.height = self.blob.height
        self.__create_starting_screen()
        self.__create_victory_condition()
        self.__choose_starting_location()
        screen.save(self.game_screen, 'start_screen')
        return
    
    def __create_victory_condition(self):
        self.victory_x, self.victory_y = self.__choose_starting_location()
        self.game_screen[self.victory_x,self.victory_y] = self.flag
        self.victory_condition = self.game_screen
        screen.save(self.victory_condition, 'victory_condition')
        return
    
    def __loss_condition(self):
        if self.game_screen(self.player_location_x, self.player_location_y) == self.tree:
            return True
        else:
            return False
        
    def __win_condition(self):
        if self.game_screen(self.player_location_x, self.player_location_y) == self.victory_condition(self.player_location_x, self.player_location_y):
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
        self.game_screen[self.player_location_x, self.player_location_y] = self.player
        return
    
    def __create_starting_screen(self):
        self.game_screen = torch.zeros((self.width, self.height), device=torch.device('cuda'), dtype=torch.int16)
        self.__create_forest()
        return
    
    def __check_valid_tree_spot(self, x, y):
        batch = []
        for i in range(-2, 2):
            for j in range(-2, 2):
                batch.append(self.game_screen(x + i, y + j))
        
        for each in batch:
            if each == self.tree:
                return False
            
        return True
    
    def __check_close_to_tree(self, x, y):
        batch = []
        for i in range(-1, 1):
            for j in range(-1, 1):
                batch.append(self.game_screen(x + i, y + j))
        
        for each in batch:
            if each == self.tree:
                return True
        return False
    
    def __create_forest(self):
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.__check_valid_tree_spot(i, j):
                    if np.random.rantint(0, 100) < 50:
                        self.game_screen[i, j] = self.tree
        return
        
    
    def __choose_starting_location(self):
        rand_x = np.random.randint(1, self.width)
        rand_y = np.random.randint(1, self.height)
        self.game_screen[rand_x, rand_y] = 255
        player_location_x = rand_x
        player_location_y = rand_y
        return
    

    def __blob_action(self, action):
        # decide on the action to take based on the input and 
        # convert that to an appropriate action for this game
        x, y = 0, 0
        if action[0] == True:
            print('x + 1')
            x += 1
        if action[0] == False:
            print('x + 0')
        if action[1] == True:
            print('x - 1')
            x += -1
        if action[1] == False:
            print('x - 0')
        if action[2] == True:
            print('y + 1')
            y += 1
        if action[2] == False:
            print('y + 0')
        if action[3] == True:
            print('y - 1')
            y += -1
        if action[3] == False:
            print('y - 0')
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
        screen.save(self.game_screen, 'start_screen')
        return
    
    # runs the game
    # if the model wins, it returns true
    # otherwise it returns false
    def play_game(self):

        # number of iterations the game will run for at max
        iter = 0
        max_iter = 1000

        # the previous action taken
        prev = 0
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
        while (self.__victory() == False):
            act = self.blob.update(self.game_screen)
            if prev == act:
                combo += 1
            else:
                combo = 0
            
            if combo > max_combo:
                break

            prev = act
            
            x, y = self.__blob_action(act)
            self.__screen_update(x, y)
            
            x = np.abs(x - self.victory_x)
            y = np.abs(y - self.victory_y)
            if (x < prev_x):
                self.blob.train(0, 255, True)
            if ( x > prev_x):
                self.blob.train(0, 255, False)
            if (y < prev_y):
                self.blob.train(1, 255, True)
            if (y > prev_y):
                self.blob.train(1, 255, False)
                
            if (self.__check_close_to_three(self.player_location_x, self.player_location_y) == True):
                self.blob.train(2, 1000, False)
            
            if (x + y < self.min_dx + self.min_dy):
                self.min_dx = x
                self.min_dy = y
            self.prev_x = x
            self.prev_y = y
            iter += 1
            print('iteration - ', iter)
            if iter > max_iter:
                break
        
        # the blob satisfies the victory condition, which is to put the white dot at the location of the victory condition
        # then we win, and we print out the personality layers

        # otherwise we lose, and just do nothing with it.
        if (self.__victory() == True):
            print("victory!")
            return True
        else:
            print("defeat!")
            screen.save(self.game_screen, 'end_screen')
        return False