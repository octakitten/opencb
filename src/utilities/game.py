import numpy as np
import torch
from cantor.src.models.camel import camel
from cantor.src.models.horse import horse
from cantor.src.utilities.screen import screen


class game():

    # the game's screen dimensions are defined here:
    width, height = 0, 0
    #the screen is here
    game_screen = 0
    def __init__(self):
        return
    
    def screen(self):
        return self.game_screen
    

'''The first game we've created.

The object is to move the white dot from a random place on the image
to the top left corner. 

This game specifically uses the camel model, and does not 
use any other models.'''
class find_food_01(game):
    # the condition the blob has to meet to win the game
    victory_condition = 0
    blob = 0

    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.starting_screen = np.zeros((self.width, self.height))
        self.game_screen = np.zeros((self.width, self.height))
        self.victory_condition = np.zeros((self.width, self.height))
        self.victory_condition[0,0] = 255
        x, y = self.choose_starting_location()
        self.game_screen[x, y] = 255
        self.blob = camel()
        return
    
    def victory(self):
        if (self.game_screen[0,0] == self.victory_condition[0,0]):
            return True
        else:
            return False
    
    def screen_update(self, x, y):
        np.roll(self.game_screen, x, axis=0)
        np.roll(self.game_screen, y, axis=1)
        return
    
    def starting_screen(self):
        self.game_screen = self.starting_screen
        x, y = self.choose_starting_location()
        self.game_screen[x, y] = 255
        return
    
    def choose_starting_location(self):
        rand_x = np.random.randint(1, self.width)
        rand_y = np.random.randint(1, self.height)
        return rand_x, rand_y
    

    def blob_action(self, action):
        # decide on the action to take based on the input and 
        # convert that to an appropriate action for this game
        x, y = 0, 0
        if action % 2 == 0:
            action = action / 2
            x += 1
        if action % 3 == 0:
            action = action / 3
            x += -1
        if action % 5 == 0:
            action = action / 5
            x += -1
        if action % 7 == 0:
            action = action / 7
            x += 1
        if action % 11 == 0:
            action = action / 11
            y += 1
        if action % 13 == 0:
            action = action / 13
            y += -1
        if action % 17 == 0:
            action = action / 17
            y += -1
        if action % 19 == 0:
            action = action / 19
            y += 1
        return x, y
    
    def play_game(self):

        # number of iterations the game will run for at max
        iter = 0
        max_iter = 1000

        # the previous action taken
        prev = 0
        # how many times the previous action has been the same as the current one
        combo = 0
        # how many times we'll allow it to combo the same action in a row
        max_combo = self.blob.width * 1.415
        
        # play the game until victory or until either combo gets too high or iterations finish
        while (self.victory() == False):
            act = self.blob.update(self.game_screen)
            if prev == act:
                combo += 1
            else:
                combo = 0
            
            if combo > max_combo:
                break

            prev = act

            x, y = self.blob_action(act)
            self.screen_update(x, y)

            iter += 1
            print('iteration - ', iter)
            if iter > max_iter:
                break
        
        # the blob satisfies the victory condition, which is to put the white dot at 0,0, the top left corner
        # then we win, and we print out the personality layers

        # otherwise we lose, and just do nothing with it.
        if (self.victory() == True):
            print("victory!")
            print('Personality layer 1')
            print(self.blob.personality1)
            print('Personality layer 2')
            print(self.blob.personality2)
            print('Personality layer 3')
            print(self.blob.personality3)
            print('Personality layer 4')
            print(self.blob.personality4)
            print('Personality layer 5')
            print(self.blob.personality5)
            print('Personality layer 6')
            print(self.blob.personality6)
            print('Personality layer 7')
            print(self.blob.personality7)
            print('Personality layer 8')
            print(self.blob.personality8)
            return self.personality1, self.personality2, self.personality3, self.personality4, self.personality5, self.personality6, self.personality7, self.personality8
        else:
            print("defeat!")
        return False
    
'''
A more generalized version of the first game.

This is the same game, where the object is to move the white dot
from a random place in a black image to the top left corner.

However, this game can use any model that takes 256x256 images as input.
It also does not work with the general model, as it sends the wrong input format.
'''
class find_food_02(game):
    # the condition the blob has to meet to win the game
    victory_condition = 0
    # the instance of horse that'll play the game
    blob = 0

    #initializing the game requires giving it a model to play with
    def __init__(self, model):
        self.blob = model
        self.width = self.blob.width
        self.height = self.blob.height
        self.__create_starting_screen()
        self.__create_victory_condition()
        screen.save(self.game_screen, 'start_screen')
        return
    
    def __create_victory_condition(self):
        self.victory_condition = torch.zeros((self.width, self.height))
        self.victory_condition[0,0] = 255
        return
    
    def __victory(self):
        if (self.game_screen[0,0] == self.victory_condition[0,0]):
            return True
        else:
            return False
    
    def __screen_update(self, x, y):
        torch.roll(self.game_screen, x, 0)
        torch.roll(self.game_screen, y, 1)
        return
    
    def __create_starting_screen(self):
        self.game_screen = torch.zeros((self.width, self.height))
        x, y = self.__choose_starting_location()
        self.game_screen[x, y] = 255
        return
    
    def __choose_starting_location(self):
        rand_x = np.random.randint(1, self.width)
        rand_y = np.random.randint(1, self.height)
        return rand_x, rand_y
    

    def __blob_action(self, action):
        # decide on the action to take based on the input and 
        # convert that to an appropriate action for this game
        x, y = 0, 0
        if action % 2 == 0:
            print('x + 1')
            x += 1
        if action % 3 == 0:
            print('x - 1')
            x += -1
        if action % 5 == 0:
            print('x - 1')
            x += -1
        if action % 7 == 0:
            print('x + 1')
            x += 1
        if action % 11 == 0:
            print('y + 1')
            y += 1
        if action % 13 == 0:
            print('y - 1')
            y += -1
        if action % 17 == 0:
            print('y - 1')
            y += -1
        if action % 19 == 0:
            print('y + 1')
            y += 1
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
        max_iter = 10000

        # the previous action taken
        prev = 0
        # how many times the previous action has been the same as the current one
        combo = 0
        # how many times we'll allow it to combo the same action in a row
        max_combo = self.blob.width * 3
        
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

            iter += 1
            print('iteration - ', iter)
            if iter > max_iter:
                break
        
        # the blob satisfies the victory condition, which is to put the white dot at 0,0, the top left corner
        # then we win, and we print out the personality layers

        # otherwise we lose, and just do nothing with it.
        if (self.__victory() == True):
            print("victory!")
            return True
        else:
            print("defeat!")
            screen.save(self.game_screen, 'end_screen')
        return False
    
    
class find_food_02(game):
    # the condition the blob has to meet to win the game
    victory_condition = 0
    # the instance of horse that'll play the game
    blob = 0

    #initializing the game requires giving it a model to play with
    def __init__(self, model):
        self.blob = model
        self.width = self.blob.width
        self.height = self.blob.height
        self.__create_starting_screen()
        self.__create_victory_condition()
        screen.save(self.game_screen, 'start_screen')
        return
    
    def __create_victory_condition(self):
        self.victory_condition = torch.zeros((self.width, self.height))
        self.victory_condition[0,0] = 255
        return
    
    def __victory(self):
        if (self.game_screen[0,0] == self.victory_condition[0,0]):
            return True
        else:
            return False
    
    def __screen_update(self, x, y):
        torch.roll(self.game_screen, x, 0)
        torch.roll(self.game_screen, y, 1)
        return
    
    def __create_starting_screen(self):
        self.game_screen = torch.zeros((self.width, self.height))
        x, y = self.__choose_starting_location()
        self.game_screen[x, y] = 255
        return
    
    def __choose_starting_location(self):
        rand_x = np.random.randint(1, self.width)
        rand_y = np.random.randint(1, self.height)
        return rand_x, rand_y
    

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
        max_iter = 10000

        # the previous action taken
        prev = 0
        # how many times the previous action has been the same as the current one
        combo = 0
        # how many times we'll allow it to combo the same action in a row
        max_combo = self.blob.width * 3
        # the previous x and y values
        prev_x = 0
        prev_y = 0
        
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
            
            if (x < prev_x):
                self.blob.train(0, 1, True)
            if ( x > prev_x):
                self.blob.train(0, 1, False)
            if (y < prev_y):
                self.blob.train(1, 1, True)
            if (y > prev_y):
                self.blob.train(1, 1, False)
            
            prev_x = x
            prev_y = y
            iter += 1
            print('iteration - ', iter)
            if iter > max_iter:
                break
        
        # the blob satisfies the victory condition, which is to put the white dot at 0,0, the top left corner
        # then we win, and we print out the personality layers

        # otherwise we lose, and just do nothing with it.
        if (self.__victory() == True):
            print("victory!")
            return True
        else:
            print("defeat!")
            screen.save(self.game_screen, 'end_screen')
        return False