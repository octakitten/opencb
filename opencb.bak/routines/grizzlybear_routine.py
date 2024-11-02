from ..utilities import find_food_02

class grizzlybear_routine():

    @staticmethod
    def run_routine():

        iters = 0
        while True:
            the_game = find_food_02(255, 255)
            if the_game.play_game() == True:
                break
            print('Retrying...')
            iters += 1
            print(iters)
        
        return the_game.blob.get_a_personality()