import numpy as np
import torch
import sys
#from pathlib import Path

#print(sys.path)
#sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#print(sys.path)

from opencb.src.opencb.models.d8a4gs import d8a4gs
from opencb.src.opencb.models.d16a1gs import d16a1gs
from opencb.src.opencb.utilities import screen
from opencb.src.opencb.models.camel import camel
from opencb.src.opencb.models.horse import horse
from opencb.src.opencb.utilities.game import find_food_01
from opencb.src.opencb.utilities.game import find_food_02
from opencb.src.opencb.routines.grizzlybear_routine import grizzlybear_routine
from opencb.src.opencb.routines.horse_routine import horse_routine
from opencb.src.opencb.models.general import general
from opencb.src.opencb.models.general_dev import general_dev
from opencb.src.opencb.utilities.game import find_food_03

def test001():
    model01 = d8a4gs()
    for i in range(100):
        input_image_fake = np.random.randint(0, 255, (256,256))
        next_action = model01.update(input_image_fake)
        print(next_action)

def test002():
    model02 = d16a1gs()
    a, b = 50, 50
    w, h = 255, 255
    for i in range(3):
        input_image_real = screen.screenshot_desktop(a, b, w, h)
        next_action = model02.update(input_image_real)
        if next_action == 2:
            print(next_action)
            a += 1
        elif next_action == 3:
            print(next_action)
            a -= 1
        else:
            print(next_action)

def test003():
    blob = camel()
    test_game = find_food_01(blob.width, blob.height)

    iter = 0
    max_iter = 1000

    prev = 0
    combo = 0
    max_combo = blob.width * 1.415
    
    while (test_game.victory() == False):
        act = blob.update(test_game.game_screen)
        print(act)
        if prev == act:
            combo += 1
        else:
            combo = 0
        
        if combo > max_combo:
            break

        prev = act

        x, y = test_game.blob_action(act)
        test_game.screen_update(x, y)

        iter += 1
        print(iter)
        if iter > max_iter:
            break
    
    if (test_game.victory() == True):
        print("victory!")
        print('Personality layer 1')
        print(blob.layer1_1_1)
        print('Personality layer 2')
        print(blob.layer1_1_2)
        print('Personality layer 3')
        print(blob.layer1_2_1)
        print('Personality layer 4')
        print(blob.layer1_2_2)
        print('Personality layer 5')
        print(blob.layer2_1_1)
        print('Personality layer 6')
        print(blob.layer2_1_2)
        print('Personality layer 7')
        print(blob.layer2_2_1)
        print('Personality layer 8')
        print(blob.layer2_2_2)
    else:
        print("defeat!")

def test004():
    the_game = find_food_01(255, 255)
    the_game.play_game()
    return

def test005():
    the_game = find_food_02(255, 255)
    the_game.play_game()
    return

def test006():
    winner = grizzlybear_routine.run_routine()
    print(winner)
    return

def test007():
    winner = horse_routine.run_routine()
    print(winner)
    return

def test008():
    model = horse()
    game = find_food_02(model)
    iters = 0
    while (game.play_game() == False):
        iters+=1
        print('game over! number of tries:')
        print(iters)
        print('restarting...')
    pers = game.blob.get_a_personality()
    print(pers)
    
def test009():
    model = general()
    model.create(255, 255, 100, 1024, 4, 2)
    game = find_food_02(model)
    iters = 0
    while (True):
        if (game.play_game() == False):
            iters+=1
            print('game over! number of tries:')
            print(iters)
            print('restarting...')
        else:
            break
        model.create(255, 255, 100, 1024, 4, 2)
        game = find_food_02(model)
    print('victory! it took this many iterations:')
    print(iters)
    print('saving to disk...')
    model.save(sys.path[0] + '/saved_models/latest.pth')
    
def test010():
    model = general()
    prev_model = 0
    model.create(255, 255, 255, 4, 4, 2)
    iters = 0
    first_attempt = True
    while (True):
        game = find_food_02(model)
        if (game.play_game() == False):
            iters+=1
            print('game over! number of tries:')
            print(iters)
            print('restarting...')
        else:
            break
        if (first_attempt):
            prev_model = model
            model.permute
        else: 
            if (model.min_dx + model.min_dy ) < (prev_model.min_dx + prev_model.min_dy):
                prev_model = model
                model.permute 
            else:
                model = prev_model
                model.permute
        first_attempt = False
    print('victory! it took this many iterations:')
    print(iters)
    print('saving to disk...')
    model.save(sys.path[0] + '/saved_models/')
    

def test011():
    model = general_dev()
    prev_model = 0
    model.create(255, 255, 255, 1000, 4, 2)
    iters = 0
    first_attempt = True
    while (True):
        game = find_food_03(model)
        if (game.play_game() == False):
            iters+=1
            print('game over! number of tries:')
            print(iters)
            print('restarting...')
        else:
            break
        if (first_attempt):
            prev_model = model
            model.permute(2,1)
        else: 
            if (model.min_dx + model.min_dy ) < (prev_model.min_dx + prev_model.min_dy):
                prev_model = model(1,2)
                model.permute 
            else:
                model = prev_model
                model.permute(2,1)
        first_attempt = False
    print('victory! it took this many iterations:')
    print(iters)
    print('saving to disk...')
    model.save(sys.path[0] + '/saved_models/')

# run test for the in development general model
# test011()