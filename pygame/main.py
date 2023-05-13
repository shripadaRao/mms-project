"""generate simple, logical ui to get coordinates and display sprits of audio source and sensor devices."""

import pygame
import random
import math


# WINDOW_SIZE = [1350, 850]
WINDOW_SIZE = [1600, 1200] #give multiple of 400 

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


# print(find_coords_dist([120,100],[130,150]))

# def sigmoid_func(dist):
#     """ takes in distance as parameter and outputs a number bw 0 and 1. and output is inversely proportianal to input(distance) """
#     output = 1 - dist**1.5/1000
#     # output = 1/dist
#     print(output)
#     return output
# sigmoid_func(100)

# def calculate_intensity(distance, initial_intensity):
#     # Ensure the distance is greater than zero
#     distance = max(distance, 0.0)
    
#     # Calculate the intensity using the inverse square law
#     intensity = initial_intensity / (distance ** 2)
#     print(intensity)
#     return intensity
# calculate_intensity(10, 1)

def vary_audio_with_distance(audio_data, distance):
    pass

def generate_audio_data_for_dist(source_audio, distance):
    """ generate varying intensity of audio data for different distances """
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
    







#***********************************    PYGAME  *************************#


pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)

#init background
background = pygame.Surface(WINDOW_SIZE)
background.fill((0,0,0))

#render background
tile_image = pygame.image.load('pygame/assets/grass-resized-sprit.png')
tree_image = pygame.image.load('pygame/assets/tree-resized-sprit.png')

num_tiles_x = WINDOW_SIZE[0] // tile_image.get_width() + 1
num_tiles_y = WINDOW_SIZE[1] // tile_image.get_height() + 1

num_of_trees = 10
# rand_tree_coords = [[random.randint(0,WINDOW_SIZE[0]), random.randint(0,WINDOW_SIZE[1])] for i in range(num_of_trees)]

def find_coord_dist(coord1, coord2):
    """ just root mean square distance """

    # dist = ((coord1[0]**2 - coord2[0]**2 ) + (coord1[1]**2 - coord2[1]**2))**0.5
    dist = math.dist(coord1, coord2)
    return dist

def generate_tree_coord():
    return [random.randint(0,WINDOW_SIZE[0]), random.randint(0,WINDOW_SIZE[1])]

def generate_tree_coords(no_of_trees):
    rand_tree_coords = []
    for i in range(no_of_trees):
        rand_tree_coords.append(generate_tree_coord())
    return rand_tree_coords

def handle_tree_generation(num_of_trees):
    safe_dist = 200 * (200 ** 0.5)
    rand_tree_coords = []

    for i in range(num_of_trees):
        rand_tree_coord = generate_tree_coord()
        count = 0
        while count < len(rand_tree_coords)-1:
            if find_coord_dist(rand_tree_coords[count], rand_tree_coord) < safe_dist:
                rand_tree_coord = generate_tree_coord()
                count = 0  # Reset the count if collision occurs
            else:
                count += 1
        rand_tree_coords.append(rand_tree_coord)
    
    return rand_tree_coords

# rand_tree_coords = handle_tree_generation(num_of_trees)
rand_tree_coords = generate_tree_coords(num_of_trees)

#handle conflict while generation of trees
# def handle_tree_generation(rand_tree_coords):
#     for i in range(len(rand_tree_coords)-1):
#         for j in range(i,len(rand_tree_coords)-1):
#             if find_coord_dist(rand_tree_coords[i], rand_tree_coords[j]) < 200:
#               rand_tree_coords[j][0], rand_tree_coords[j][1] = rand_tree_coords[j][0] + 200, rand_tree_coords[j][1] + 200
#     return rand_tree_coords

# def handle_tree_collisions(rand_tree_coords):
#     safe_dist = 200*(200**0.5)

#     def is_safe_coords(coord1, coord2):
#         if find_coord_dist(coord1, coord2) > safe_dist:
#             return True
#         else:
#             return False

#     for i in range(len(rand_tree_coords)):

#     while is_safe_coords() is False:



RESET_BUTTON_DIMS = [100,50] 
reset_button_coords = [1150, 50]

RESET_BUTTON_DIMS= [WINDOW_SIZE[0]/13.5, WINDOW_SIZE[1]/17]
reset_button_coords = [WINDOW_SIZE[0]/1.174, WINDOW_SIZE[1]/17]
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()

            #below is condition for checking whether the user has clicked within the dims of button
            #reset button true
            if reset_button_coords[0] <= mouse_pos[0] <= reset_button_coords[0]+RESET_BUTTON_DIMS[0] and reset_button_coords[1] <= mouse_pos[1] <= reset_button_coords[1]+RESET_BUTTON_DIMS[1]:
                
                audio_source_coords = generate_audio_source_coords()
                devices_coords = generate_device_coords()
                # print("audio source coords: ",audio_source_coords)
                # print('device coords[0]', devices_coords[0])

                # print("distance : ",find_coords_dist(audio_source_coords, devices_coords[0]))

                #for trees randomly generate some coordinate in range of WINDOW_SIZE
                rand_tree_coords = [[random.randint(0,WINDOW_SIZE[0]), random.randint(0,WINDOW_SIZE[1])] for i in range(num_of_trees)]
                # rand_tree_coords = handle_tree_generation(rand_tree_coords)
                pygame.display.flip()

    screen.blit(background, (0,0))

    #render background
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate the coordinates for each tile
            tile_x = x * tile_image.get_width()
            tile_y = y * tile_image.get_height()

            screen.blit(tile_image, (tile_x, tile_y))

    for tree_coord in rand_tree_coords:
        x,y = tree_coord[0], tree_coord[1]
        #replace circle with tree sprit
        # pygame.draw.circle(screen, (0,0,0), (x,y), 10)
        screen.blit(tree_image, (x,y))


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