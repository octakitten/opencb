'''import torch
from opencb import models
from opencb import games'''
from opencb import routines

#routines.iteration.test014('/indev-tests')
'''import sys


def main():

    model = models.general_dev()
    prev = None
    model.create(255, 255, 255, 2000, 4, 2)
    iters = 0
    first_attempt = True
    while (True):
        game = games.find_food_03(model)
        if (game.play_game() == False):
            iters+=1
            print('game over! number of tries:')
            print(iters)
            print('restarting...')
        else:
            break
        if (first_attempt):
            prev_model = model
            model.permute(2, 1)
        else:
            if (model.min_dx + model.min_dy) < (prev_model.min_dx + prev_model.min_dy):
                prev_model = model(1, 2)
                model.permute
            else:
                model = prev_model
                model.permute(2, 1)
        first_attempt = False
        print('victory! it took this many iterations:')
        print(iters)
        print('saving to disk...')
        model.save(sys.path[0] + '/saved_models/')
    return
'''

routines.iteration.run_general_dev()
