import torch

class game001():

    screen = 0
    victory = 0

    def __init__(self):
        return
    
    def __create_victory_condition(self):
        self.victory = torch.zeros((255,255))
        self.victory[0,0] = 255
        return
