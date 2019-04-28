# Interfaces or the classes in the snake game implementation
# By adhering to the interfaces we can more easily share code
# and more easily unit test
# To install interface support: pip install python-interface

# This interface is based on snake1 2019-04-28
# Last modified 2019-04-28  kja

from interface import implements, Interface


class ICube(Interface, object):
    def __init__(self, start, dirnx=1, dirny=0, color=(255, 0, 0)):
        pass
    
    def move(self, dirnx, dirny):
        pass
    
    def draw(self, surface, eyes=False):
        pass


class ISnake(Interface, object):
    def __init__(self, color, pos):
        pass
    
    def move(self, action = None):
        pass
    
    def move_gym(self, action):
        pass
    
    def reset(self, pos):
        pass
    
    def addCube(self):
        pass
    
    def draw(self, surface):
        pass

class IEnv(Interface):
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def step(self, action):
        pass

