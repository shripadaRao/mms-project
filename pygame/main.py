"""generate simple, logical ui to get coordinates and display sprits of audio source and sensor devices."""

import pygame
import random
import math
import requests
import json
import librosa
import numpy as np


WINDOW_SIZE = [1600, 1000] #give multiple of 200 

# AUDIO_SOURCE_VARIATION = [60,60] #width, height
# DEVICE_COORD_VARIATION = [60,60] #width, height

AUDIO_SOURCE_VARIATION = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]
DEVICE_COORD_VARIATION = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]

AUDIO_CLASSIFICATION_URL = 'http://127.0.0.1:5000/classify-audio-data/'

SENSOR_SPRIT_PATH = "pygame/assets/sensor-sprit.png"
SAWING_SPRIT_PATH = "pygame/assets/saw-resized-sprite.png"
AMBIENT_SOUND_PATH = 'pygame/assets/Ambient-sound-sprite.png'

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


def mult_factor(dist):
    """ takes in distance as parameter and outputs a number bw 0 and 1. and output is inversely proportianal to input(distance) """
    MAX_DIST_BW_AUDIO_DEVICE = 670
    output = 1 - dist/MAX_DIST_BW_AUDIO_DEVICE

    return output

def generate_audio_data_for_dist(source_audio, distance):
    """ generate varying intensity of audio data for different distances """
    source_audio = source_audio * mult_factor(distance)
    return source_audio

menu_isAmbient = True
menu_isSawing = False

get_audio_source_sprite = lambda: AMBIENT_SOUND_PATH if menu_isAmbient else SAWING_SPRIT_PATH


class AudioSource:
    """ should be able to run audio continously, combo of both ambient and sawing(min sawing time > 2 sec) 
        should be able to vary audio with distance. 
        should be able to display some sprit and 'react' when audio is actually sawing. 
    """
    def __init__(self, coords) -> None:
        self.coords = coords
        self.isSawing = False
        self.load_asset()
    
    def load_asset(self, source_path=get_audio_source_sprite()):
        # pygame.draw.circle(screen, (0, 0, 200), (self.coords[0], self.coords[1]), 25)
        saw_tile = pygame.image.load(source_path)
        screen.blit(saw_tile,(self.coords[0], self.coords[1]))



class SensorDevice:
    """ should be able to listen to incoming audio, send request the same to server for prediction. 
        should be able to display sprit and 'react' when audio is detected.  
    """
    #tradeoff - vary audio in this class than AudioSource.
    def __init__(self, coords) -> None:
        self.coords = coords
        self.isSawing = False   #use pygame.display.update() instead of display.flip()
        self.load_asset()

    def load_asset(self, source_path=SENSOR_SPRIT_PATH): 
        pygame.draw.circle(screen, (0, 0, 250), (self.coords[0], self.coords[1]), 25)
        # sensor_tile = pygame.image.load(source_path)
        # screen.blit(sensor_tile,(self.coords[0], self.coords[1]))

    def recieve_audio_data(self, original_audio_data, distance_bw_audio_source_sensor):
        return generate_audio_data_for_dist(original_audio_data, distance_bw_audio_source_sensor)
    
    def predict_audio_data(self, recieved_audio_data):
        # r = requests.get('http://127.0.0.1:5000/')
        # resp = r.text

        r = requests.post(AUDIO_CLASSIFICATION_URL, data=json.dumps({"audio_data":recieved_audio_data.tolist()}), headers = {"Content-Type": "application/json"})
        return r.json()
    
    def relocate_coords(self, coords):
        self.coords = coords


# sensor = SensorDevice([])
# y, sr = librosa.load('test/10s_ambient.wav')
# sensor.predict_audio_data(y)


#***********************************    PYGAME  *************************#

NUM_OF_TREES = 15


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

# num_of_trees = 10
# rand_tree_coords = [[random.randint(0,WINDOW_SIZE[0]), random.randint(0,WINDOW_SIZE[1])] for i in range(num_of_trees)]

def find_coord_dist(coord1, coord2):
    # just root mean square distance 
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
    safe_dist = 200 * (2**0.5)
    rand_tree_coords = []

    for i in range(num_of_trees):
        rand_tree_coord = generate_tree_coord()
        count = 0
        while count < len(rand_tree_coords):
            if find_coord_dist(rand_tree_coords[count], rand_tree_coord) < safe_dist:
                rand_tree_coord = generate_tree_coord()
                count = 0  # Reset the count if collision occurs
            else:
                count += 1
        rand_tree_coords.append(rand_tree_coord)
    
    return rand_tree_coords

rand_tree_coords = handle_tree_generation(NUM_OF_TREES)



# RESET_BUTTON_DIMS = [100,50] 
# reset_button_coords = [1150, 50]

RESET_BUTTON_DIMS= [WINDOW_SIZE[0]/13.5, WINDOW_SIZE[1]/20]
reset_button_coords = [WINDOW_SIZE[0]/1.1799, WINDOW_SIZE[1]/50]

RUN_BUTTON_DIMS = [WINDOW_SIZE[0]/20, WINDOW_SIZE[1]/20]
run_button_coords = [WINDOW_SIZE[0]/1.081, WINDOW_SIZE[1]/50]

DROP_DOWN_BUTTON_DIMS = [WINDOW_SIZE[0]/60,WINDOW_SIZE[1]/20]
drop_button_coords = [WINDOW_SIZE[0]/1.020, WINDOW_SIZE[1]/50]


audio_source = AudioSource(audio_source_coords)
sensor_list = SensorDevice(devices_coords[0]), SensorDevice(devices_coords[1]), SensorDevice(devices_coords[2]), SensorDevice(devices_coords[3])
[sensor1, sensor2, sensor3, sensor4] = sensor_list

menu_isVisible = False
menu_width, menu_height = 130, 60
menu_contents = ['sawing', 'ambient']

menu_isAmbient = True
menu_isSawing = False

radio_button_radius = 7

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()

            if reset_button_coords[0] <= mouse_pos[0] <= reset_button_coords[0]+RESET_BUTTON_DIMS[0] and reset_button_coords[1] <= mouse_pos[1] <= reset_button_coords[1]+RESET_BUTTON_DIMS[1]:
                #if reset button is true, run below block
                audio_source_coords = generate_audio_source_coords()
                devices_coords = generate_device_coords()

                rand_tree_coords = handle_tree_generation(NUM_OF_TREES)
                sensor1.isSawing = False
                sensor2.isSawing = False
                sensor3.isSawing = False
                sensor4.isSawing = False
                pygame.display.flip()


            if run_button_coords[0] <= mouse_pos[0] <= run_button_coords[0]+RUN_BUTTON_DIMS[0] and run_button_coords[1] <= mouse_pos[1] <= run_button_coords[1]+RUN_BUTTON_DIMS[1]:
                # print('run clicked!')
                audio_filepath = 'test/10s_sawing.wav'
                source_audio_data, sample_rate = librosa.load(audio_filepath)

                recieved_audio_data_all_sensors = []    
                for idx, sensor in enumerate(sensor_list):
                    recieved_audio_data =sensor.recieve_audio_data(source_audio_data, find_coord_dist(sensor.coords,audio_source.coords))
                    recieved_audio_data_all_sensors.append(recieved_audio_data)

                pred = sensor1.predict_audio_data(recieved_audio_data_all_sensors[0])
                # print('prediction: ', pred)
                # pred['sawing'] = 1

                # if 'sawing' in audio_filepath:
                #     sensor1.isSawing = True
                #     sensor2.isSawing = True
                #     sensor3.isSawing = True
                #     sensor4.isSawing = True
                    
                #     print('SAWING!')
                # else:
                #     print('no sawing!')
                if menu_isSawing:
                    sensor1.isSawing = True
                    sensor2.isSawing = True
                    sensor3.isSawing = True
                    sensor4.isSawing = True
                                
            if drop_button_coords[0] <= mouse_pos[0] <= drop_button_coords[0]+DROP_DOWN_BUTTON_DIMS[0] and drop_button_coords[1] <= mouse_pos[1] <= drop_button_coords[1]+DROP_DOWN_BUTTON_DIMS[1]:
                menu_isVisible = not menu_isVisible

            if menu_isVisible and drop_button_coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= drop_button_coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and drop_button_coords[1]+45+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= drop_button_coords[1]+45+(menu_height)//len(menu_contents) + radio_button_radius +10:
                menu_isSawing = True
                menu_isAmbient = False

            if menu_isVisible and drop_button_coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= drop_button_coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and drop_button_coords[1]+75+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= drop_button_coords[1]+75+(menu_height)//len(menu_contents) + radio_button_radius +10:
                menu_isSawing = False
                menu_isAmbient = True      


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
        screen.blit(tree_image, (x,y))

    
    #menu drop down
    if menu_isVisible:
        pygame.draw.rect(screen,(255,255,255),(drop_button_coords[0]-100, drop_button_coords[1]+60,menu_width,menu_height))
        for i,item in enumerate(menu_contents):
            text = font.render(item, True, (0,0,0))
            text_rect = text.get_rect(center=(drop_button_coords[0]-115 + menu_width // 2, drop_button_coords[1]+60 + (menu_height // len(menu_contents)) * i + menu_height // (2 * len(menu_contents))))
            screen.blit(text, text_rect)

        if menu_isSawing:
            pygame.draw.circle(screen, (0,0,0), (drop_button_coords[0]-50 + menu_width // 2, drop_button_coords[1]+45+(menu_height)//len(menu_contents)),radio_button_radius)
        if menu_isAmbient:
            pygame.draw.circle(screen, (0,0,0), (drop_button_coords[0]-50 + menu_width // 2, drop_button_coords[1]+75+(menu_height)//len(menu_contents)),radio_button_radius)

    #audio source
    # pygame.draw.circle(screen, (0, 0, 200), (audio_source_coords[0], audio_source_coords[1]), 25)
    audio_source = AudioSource(audio_source_coords)
    audio_source.load_asset()

    # #devices
    # [bl,tl, tr, br]=devices_coords
    # pygame.draw.circle(screen, (250, 0, 0), (bl[0], bl[1]), 25)
    # pygame.draw.circle(screen, (250, 0, 0), (tl[0], tl[1]), 25)
    # pygame.draw.circle(screen, (250, 0, 0), (tr[0], tr[1]), 25)
    # pygame.draw.circle(screen, (250, 0, 0), (br[0], br[1]), 25)
    # sensor1, sensor2, sensor3, sensor4 = (devices_coords[0]), SensorDevice(devices_coords[1]), SensorDevice(devices_coords[2]), SensorDevice(devices_coords[3])
    sensor1.relocate_coords(devices_coords[0])
    sensor2.relocate_coords(devices_coords[1])
    sensor3.relocate_coords(devices_coords[2])
    sensor4.relocate_coords(devices_coords[3])
 
    sensor1.load_asset()
    sensor2.load_asset()
    sensor3.load_asset()
    sensor4.load_asset()

    audio_source.load_asset(get_audio_source_sprite())


    if sensor1.isSawing:
        pygame.draw.circle(screen, (250, 0, 0), (devices_coords[0][0], devices_coords[0][1]), 35, 5)

    if sensor2.isSawing:
        pygame.draw.circle(screen, (250, 0, 0), (devices_coords[1][0], devices_coords[1][1]), 35, 5)

    if sensor3.isSawing:
        pygame.draw.circle(screen, (250, 0, 0), (devices_coords[2][0], devices_coords[2][1]), 35, 5)

    if sensor4.isSawing:
        pygame.draw.circle(screen, (250, 0, 0), (devices_coords[3][0], devices_coords[3][1]), 35, 5)


    pygame.draw.rect(screen, (255,0,0), (reset_button_coords[0], reset_button_coords[1], RESET_BUTTON_DIMS[0], RESET_BUTTON_DIMS[1]))
    font = pygame.font.Font(None, 36)    
    reset_text = font.render("Reset", True, (255, 255, 255))
    text_rect = reset_text.get_rect(center=(reset_button_coords[0] + RESET_BUTTON_DIMS[0] // 2, reset_button_coords[1] + RESET_BUTTON_DIMS[1] // 2))
    screen.blit(reset_text, text_rect)  

    pygame.draw.rect(screen, (255,0,0), (run_button_coords[0], run_button_coords[1], RUN_BUTTON_DIMS[0], RUN_BUTTON_DIMS[1]))
    font = pygame.font.Font(None, 36)    
    run_text = font.render("Run", True, (255, 255, 255))
    text_rect = reset_text.get_rect(center=(run_button_coords[0] + RUN_BUTTON_DIMS[0] // 2, run_button_coords[1] + RUN_BUTTON_DIMS[1] // 2))
    screen.blit(run_text, text_rect) 


    get_dropdown_symbol = lambda: "^" if menu_isVisible else "v"

    pygame.draw.rect(screen, (255,0,0), (drop_button_coords[0], drop_button_coords[1], DROP_DOWN_BUTTON_DIMS[0], DROP_DOWN_BUTTON_DIMS[1]))
    font = pygame.font.Font(None, 36)    
    drop_text = font.render(get_dropdown_symbol(), True, (255, 255, 255))
    text_rect = reset_text.get_rect(center=((drop_button_coords[0] + DROP_DOWN_BUTTON_DIMS[0] // 2)+25, drop_button_coords[1] + DROP_DOWN_BUTTON_DIMS[1] // 2))
    screen.blit(drop_text, text_rect) 

    pygame.display.flip()

pygame.quit()