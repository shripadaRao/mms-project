"""generate simple, logical ui to get coordinates and display sprits of audio source and sensor devices."""

import pygame
import random


# WINDOW_SIZE = [1350, 850]
# WINDOW_SIZE = [1600, 800]
# WINDOW_SIZE = [1500,1000]
WINDOW_SIZE = [1700, 1000]

# AUDIO_SOURCE_VARIATION = [60,60] #width, height
# DEVICE_COORD_VARIATION = [60,60] #width, height

AUDIO_SOURCE_VARIATION = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]
DEVICE_COORD_VARIATION = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]


#UTIL functions
def generate_audio_source_coords(): # returns [width, height]
    """ audio source is likely to be around center of the screen """
    window_center = [i/2 for i in WINDOW_SIZE]
    variation_num = [random.randint(- AUDIO_SOURCE_VARIATION[0], AUDIO_SOURCE_VARIATION[0]), random.randint(- AUDIO_SOURCE_VARIATION[1], AUDIO_SOURCE_VARIATION[1])]
    return [window_center[0]+variation_num[0], window_center[1]+variation_num[1]]

audio_source_coords = (generate_audio_source_coords())


def generate_device_coords(devices_number=4):
    """ come up with an algo to swarm the screen with positions according to number of devices """
    #hard code for 4 devices for now
    window_center = [i/2 for i in WINDOW_SIZE]
    variation_num = [random.randint(- AUDIO_SOURCE_VARIATION[0], AUDIO_SOURCE_VARIATION[0]), random.randint(- AUDIO_SOURCE_VARIATION[1], AUDIO_SOURCE_VARIATION[1])]
    variation_num = [i*1.5 for i in variation_num]

    bottom_left = [window_center[0]- (window_center[0]/2) + variation_num[0], window_center[1]- (window_center[1]/2) + variation_num[1]]
    variation_num = [random.randint(- AUDIO_SOURCE_VARIATION[0], AUDIO_SOURCE_VARIATION[0]), random.randint(- AUDIO_SOURCE_VARIATION[1], AUDIO_SOURCE_VARIATION[1])]
    variation_num = [i*1.5 for i in variation_num]


    top_left = [window_center[0]- (window_center[0]/2) + variation_num[0], window_center[1]+ (window_center[1]/2) + variation_num[1]]
    variation_num = [random.randint(- AUDIO_SOURCE_VARIATION[0], AUDIO_SOURCE_VARIATION[0]), random.randint(- AUDIO_SOURCE_VARIATION[1], AUDIO_SOURCE_VARIATION[1])]
    variation_num = [i*1.5 for i in variation_num]

    bottom_right = [window_center[0]+ (window_center[0]/2) + variation_num[0], window_center[1]- (window_center[1]/2) + variation_num[1]]
    variation_num = [random.randint(- AUDIO_SOURCE_VARIATION[0], AUDIO_SOURCE_VARIATION[0]), random.randint(- AUDIO_SOURCE_VARIATION[1], AUDIO_SOURCE_VARIATION[1])]
    variation_num = [i*1.5 for i in variation_num]

    top_right = [window_center[0]+ (window_center[0]/2) + variation_num[0], window_center[1]+ (window_center[1]/2) + variation_num[1]]

    return [bottom_left, top_left, top_right, bottom_right]
    
devices_coords = generate_device_coords()


def find_coords_dist(coord1, coord2):
    """ just root mean square distance """
    #coord1 = [120,100] coord2 = [130,150]
    dist = []
    for i in range(len(coord1)):
        ele_dist = (coord2[i]**2 - coord1[i]**2)**0.5
        dist.append(ele_dist)
    dist = sum(dist)
    return dist

# print(find_coords_dist([120,100],[130,150]))

# def vary_audio_with_distance(audio_data, distance):
#     pass

# def generate_audio_data_for_dist(source_audio, distance):
#     """ generate varying intensity of audio data for different distances """
#     pass


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
    






#***********************************    PYGAME  *************************#
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
# WINDOW_SIZE = [1350, 850]


RESET_BUTTON_DIMS = [100,50] 
reset_button_coords = [1150, 50]

RESET_BUTTON_DIMS= [WINDOW_SIZE[0]/13.5, WINDOW_SIZE[1]/17]
reset_button_coords = [WINDOW_SIZE[0]/1.174, WINDOW_SIZE[1]/17]
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            # below is condition for checking whether the user has clicked within the dims of button
            if reset_button_coords[0] <= mouse_pos[0] <= reset_button_coords[0]+RESET_BUTTON_DIMS[0] and reset_button_coords[1] <= mouse_pos[1] <= reset_button_coords[1]+RESET_BUTTON_DIMS[1]:
                #reset button true
                audio_source_coords = generate_audio_source_coords()
                devices_coords = generate_device_coords()

                pygame.display.flip()


    screen.fill((255, 255, 255))

    #audio source
    pygame.draw.circle(screen, (0, 0, 200), (audio_source_coords[0], audio_source_coords[1]), 25)

    #devices
    [bl,tl, tr, br]=devices_coords
    pygame.draw.circle(screen, (100, 100, 0), (bl[0], bl[1]), 25)
    pygame.draw.circle(screen, (100, 100, 0), (tl[0], tl[1]), 25)
    pygame.draw.circle(screen, (100, 100, 0), (tr[0], tr[1]), 25)
    pygame.draw.circle(screen, (100, 100, 0), (br[0], br[1]), 25)

    pygame.draw.rect(screen, (255,0,0), (reset_button_coords[0], reset_button_coords[1], RESET_BUTTON_DIMS[0], RESET_BUTTON_DIMS[1]))
    font = pygame.font.Font(None, 36)    
    text = font.render("Reset", True, (255, 255, 255))
    text_rect = text.get_rect(center=(reset_button_coords[0] + RESET_BUTTON_DIMS[0] // 2, reset_button_coords[1] + RESET_BUTTON_DIMS[1] // 2))
    screen.blit(text, text_rect)   
    pygame.display.flip()

pygame.quit()