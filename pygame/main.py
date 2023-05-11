"""generate simple, logical ui to get coordinates and display sprits of audio source and sensor devices."""

import pygame


WINDOW_SIZE = [1350, 850]


#UTIL functions
def generate_audio_source_coords():
    """ audio source is likely to be around center of the screen """
    pass

def generate_device_coords(devices_number=4):
    """ come up with an algo to swarm the screen with positions according to number of devices """
    #hard code for 4 devices for now
    pass


def find_coords_dist(coord1, coord2):
    """ just root mean square distance """
    pass

# def vary_audio_with_distance(audio_data, distance):
#     pass

# def generate_audio_data_for_dist(source_audio, distance):
#     """ generate varying intensity of audio data for different distances """
#     pass


def coords_handler(audio_coords, devices_coords):
    """ to check validity and correct any conflicts in coordinates generation conflicts """
    pass




class AudioSource:
    """ should be able to run audio continously, combo of both ambient and sawing(min sawing time > 2 sec) 
        should be able to vary audio with distance. 
        should be able to display some sprit and 'react' when audio is actually sawing. 
    """
    def __init__(self, coords) -> None:
        self.coords = coords
        self.isSawing = False
    
    def load_asset(self, source_path):
        pass



class SensorDevice:
    """ should be able to listen to incoming audio, send request the same to server for prediction. 
        should be able to display sprit and 'react' when audio is detected.  
    """
    #tradeoff - vary audio in this class than AudioSource.
    def __init__(self, coords) -> None:
        self.coords = coords
        self.isSawing = False   #use pygame.display.update() instead of display.flip()

    def load_asset(self, source_path): #will also be used to update display
        pass

    def predict_audio_data(self, audio_data): #handle api interaction here and set flag
        pass
    


pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
    pygame.display.flip()

pygame.quit()