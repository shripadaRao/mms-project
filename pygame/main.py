"""generate simple, logical ui to get coordinates and display sprits of audio source and sensor devices."""

import pygame
import random
import math
import requests
import json
import librosa
import numpy as np
import asyncio
import aiohttp
import time
import gzip


WINDOW_SIZE = [1600, 1000] #give multiple of 200 

# AUDIO_SOURCE_VARIATION = [60,60] #width, height
# DEVICE_COORD_VARIATION = [60,60] #width, height

AUDIO_SOURCE_VARIATION = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]
DEVICE_COORD_VARIATION = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]

AUDIO_CLASSIFICATION_API = 'http://127.0.0.1:5000/classify-audio-data/'
TDOA_ESTIMATION_API = "http://127.0.0.1:5000/localize-audio-source/tdoa/"

SENSOR_SPRITE_PATH = "pygame/assets/sensor-sprit.png"
SAWING_SPRITE_PATH = "pygame/assets/saw-resized-sprite.png"
AMBIENT_SPRITE_PATH = 'pygame/assets/Ambient-sound-sprite.png'

PYGAME_SAWING_AUDIO_FILEPATH = "pygame/audio_files/background_sawing.wav"
SAWING_AUDIO_FILEPATH = "pygame/audio_files/merged_sawing_ambient.wav"
AMBIENT_AUDIO_FILEPATH = "pygame/audio_files/10s_ambient.wav"


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

get_audio_source_sprite = lambda: AMBIENT_SPRITE_PATH if menu_isAmbient else SAWING_SPRITE_PATH


class AudioSource:
    """ should be able to run audio continously, combo of both ambient and sawing(min sawing time > 2 sec) 
        should be able to vary audio with distance. 
        should be able to display some sprite and 'react' when audio is actually sawing. 
        should be able to generate/create audio loop, should be able to stream the same as well.
    """
    def __init__(self, coords) -> None:
        self.coords = coords
        self.isSawing = False
        self.load_asset()
    
    def load_asset(self, source_path=get_audio_source_sprite()):
        # pygame.draw.circle(screen, (0, 0, 200), (self.coords[0], self.coords[1]), 25)
        saw_tile = pygame.image.load(source_path)
        screen.blit(saw_tile,(self.coords[0]-100, self.coords[1]-100))

    def create_audio_source_loop(self, source_audio_clips_list, isSawing, BUFFER_FRAME_SIZE = 2, BUFFER_MAX_FRAMES = 3): #buffer frame size is the 
        #isSawing then, add sawing audio clips randomly to the queue.
        
        pass

    def stream_audio_data(self, ):
        pass



class SensorDevice:
    """ should be able to listen to incoming audio, send request the same to server for prediction. 
        should be able to display sprite and 'react' when audio is detected.  
    """
    #tradeoff - vary audio in this class than AudioSource.
    def __init__(self, coords) -> None:
        self.coords = coords
        self.isSawing = False   #use pygame.display.update() instead of display.flip()
        self.load_asset()

    def load_asset(self, source_path=SENSOR_SPRITE_PATH): 
        pygame.draw.circle(screen, (0, 0, 250), (self.coords[0], self.coords[1]), 25)
        # sensor_tile = pygame.image.load(source_path)
        # screen.blit(sensor_tile,(self.coords[0], self.coords[1]))

    def recieve_audio_data(self, original_audio_data, distance_bw_audio_source_sensor):
        return generate_audio_data_for_dist(original_audio_data, distance_bw_audio_source_sensor)
    
    def recieve_streaming_audio(self, recieved_audio_data):
        pass
    
    def predict_audio_data(self, recieved_audio_data):
        r = requests.post(AUDIO_CLASSIFICATION_API, data=json.dumps({"audio_data":recieved_audio_data.tolist()}), headers = {"Content-Type": "application/json"})
        return r.json()
    
    async def async_predict_audio_data(self, session, received_audio_data):
        # Compress the audio data using gzip
        compressed_audio_data = gzip.compress(received_audio_data.tobytes())

        headers = {'Content-Encoding': 'gzip'}

        async with session.post(AUDIO_CLASSIFICATION_API, data=compressed_audio_data, headers=headers) as response:
            # Decode the response if needed
            result = await response.json()
            return result
        
    
    def relocate_coords(self, coords):
        self.coords = coords


def sensors_data_tdoa(sensors_coords, distances_arr):
    sensors_data = [
        {
        "sensor1" : {
            "coords" : sensors_coords[0],
            "distance_to_audio_source" : distances_arr[0]
            }
        },
        {
        "sensor2" : {
            "coords" : sensors_coords[1],
            "distance_to_audio_source" : distances_arr[1]
            }
        },
            {
        "sensor3" : {
            "coords" : sensors_coords[2],
            "distance_to_audio_source" : distances_arr[2]
            }
        },
            {
        "sensor4" : {
            "coords" : sensors_coords[3],
            "distance_to_audio_source" : distances_arr[3]
            }
        }	
    ]
    return sensors_data
     
def estimate_tdoa(sensors_data):
    response = requests.post(TDOA_ESTIMATION_API, json=sensors_data)
    if response.status_code == 200:
        tdoa_estimate = response.json()
        return tdoa_estimate
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None




#***********************************    PYGAME  *************************#


#load audio_files

ambient_audio_data, sample_rate = librosa.load(AMBIENT_AUDIO_FILEPATH)
sawing_audio_data, sample_rate = librosa.load(SAWING_AUDIO_FILEPATH)


#display time
def current_time_string():
    return time.strftime("%H:%M:%S", time.localtime())

#play audio
def play_audio(audio_file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play(-1)

def handle_background_audio( playing_audio):
    if source_audio_file_path != playing_audio:
        pygame.mixer.music.stop()
        pygame.mixer.init()
        pygame.mixer.music.load(source_audio_file_path)
        pygame.mixer.music.play(-1)
    else:
        pass



# source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else SAWING_AUDIO_FILEPATH
# source_audio_file_path = "audio_files/background_sawing.wav" if menu_isAmbient else SAWING_AUDIO_FILEPATH
source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else "audio_files/background_sawing.wav"
source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else PYGAME_SAWING_AUDIO_FILEPATH



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

display_time_coords = [WINDOW_SIZE[0]/2.15, WINDOW_SIZE[1]/50]



audio_source = AudioSource(audio_source_coords)
sensor_list = SensorDevice(devices_coords[0]), SensorDevice(devices_coords[1]), SensorDevice(devices_coords[2]), SensorDevice(devices_coords[3])
[sensor1, sensor2, sensor3, sensor4] = sensor_list

menu_isVisible = False
menu_width, menu_height = 140, 90
menu_contents = ['sawing', 'ambient','tdoa']

menu_isAmbient = True
menu_isSawing = False

radio_button_radius = 7

is_tdoa = False

running = True

# print(s)
play_audio(source_audio_file_path)
playing_audio = source_audio_file_path

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

                source_audio_data = ambient_audio_data if menu_isAmbient else sawing_audio_data

                recieved_audio_data_all_sensors = []    
                for idx, sensor in enumerate(sensor_list):
                    recieved_audio_data =sensor.recieve_audio_data(source_audio_data, find_coord_dist(sensor.coords,audio_source.coords))
                    recieved_audio_data_all_sensors.append(recieved_audio_data)

                async def gather_pred_data_async():

                    async def main():
                        async with aiohttp.ClientSession() as session:
                            responses = await asyncio.gather(
                                sensor1.async_predict_audio_data(session, recieved_audio_data_all_sensors[0]),
                                sensor2.async_predict_audio_data(session, recieved_audio_data_all_sensors[1]),
                                sensor3.async_predict_audio_data(session, recieved_audio_data_all_sensors[2]),
                                sensor3.async_predict_audio_data(session, recieved_audio_data_all_sensors[3])
                            )

                            return responses

                    result = await main()
                    return result

                devices_predictions = asyncio.run(gather_pred_data_async())  
                print(devices_predictions)


                if is_tdoa:
                    sensor_source_dist_arr = []
                    for device_coord in devices_coords:
                        sensor_source_dist_arr.append(find_coord_dist(device_coord,audio_source_coords))
                    sensors_data = sensors_data_tdoa(devices_coords,sensor_source_dist_arr) 
                    tdoa_est = estimate_tdoa(sensors_data)
                    
                    print("actual coords: ", audio_source_coords)            
                    print("tdoa prediction: ", tdoa_est)       
             
                sensor1.isSawing = True if 'sawing' in devices_predictions[0] else sensor1.isSawing
                sensor2.isSawing = True if 'sawing' in devices_predictions[1] else sensor2.isSawing
                sensor3.isSawing = True if 'sawing' in devices_predictions[2] else sensor3.isSawing
                sensor4.isSawing = True if 'sawing' in devices_predictions[3] else sensor4.isSawing

        
            if drop_button_coords[0] <= mouse_pos[0] <= drop_button_coords[0]+DROP_DOWN_BUTTON_DIMS[0] and drop_button_coords[1] <= mouse_pos[1] <= drop_button_coords[1]+DROP_DOWN_BUTTON_DIMS[1]:
                menu_isVisible = not menu_isVisible

            if menu_isVisible and drop_button_coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= drop_button_coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and drop_button_coords[1]+45+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= drop_button_coords[1]+45+(menu_height)//len(menu_contents) + radio_button_radius +10:
                menu_isSawing = True
                menu_isAmbient = False

            if menu_isVisible and drop_button_coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= drop_button_coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and drop_button_coords[1]+75+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= drop_button_coords[1]+75+(menu_height)//len(menu_contents) + radio_button_radius +10:
                menu_isSawing = False
                menu_isAmbient = True      

            if menu_isVisible and drop_button_coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= drop_button_coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and drop_button_coords[1]+105+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= drop_button_coords[1]+105+(menu_height)//len(menu_contents) + radio_button_radius +10:
                # menu_isSawing = False
                # menu_isAmbient = True 
                is_tdoa = not is_tdoa


    screen.blit(background, (0,0))

    source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else "audio_files/background_sawing.wav"
    source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else PYGAME_SAWING_AUDIO_FILEPATH

    handle_background_audio(playing_audio)
    # play_audio(source_audio_file_path())
    playing_audio = source_audio_file_path

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
        if is_tdoa:
            pygame.draw.circle(screen, (0,0,0), (drop_button_coords[0]-50 + menu_width // 2, drop_button_coords[1]+105+(menu_height)//len(menu_contents)),radio_button_radius)


    #audio source
    # pygame.draw.circle(screen, (200, 00, 00), (audio_source_coords[0], audio_source_coords[1]), 25)
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
    # font = pygame.font.Font(None, 36)    
    run_text = font.render("Run", True, (255, 255, 255))
    text_rect = reset_text.get_rect(center=(run_button_coords[0] + RUN_BUTTON_DIMS[0] // 2, run_button_coords[1] + RUN_BUTTON_DIMS[1] // 2))
    screen.blit(run_text, text_rect) 


    get_dropdown_symbol = lambda: "^" if menu_isVisible else "v"

    pygame.draw.rect(screen, (255,0,0), (drop_button_coords[0], drop_button_coords[1], DROP_DOWN_BUTTON_DIMS[0], DROP_DOWN_BUTTON_DIMS[1]))
    # font = pygame.font.Font(None, 36)    
    drop_text = font.render(get_dropdown_symbol(), True, (255, 255, 255))
    text_rect = reset_text.get_rect(center=((drop_button_coords[0] + DROP_DOWN_BUTTON_DIMS[0] // 2)+25, drop_button_coords[1] + DROP_DOWN_BUTTON_DIMS[1] // 2))
    screen.blit(drop_text, text_rect) 


    #display time
    time_string = current_time_string()
    font = pygame.font.Font(None,40)
    text_surface = font.render(time_string, True, (255,255,255))
    screen.blit(text_surface,(display_time_coords[0], display_time_coords[1]))


    #tdoa
    #draw circle of radius equal to distance between source and sensor at the sensor
    if is_tdoa:
        pygame.draw.circle(screen, (250, 0, 50), (devices_coords[0][0], devices_coords[0][1]), find_coord_dist([devices_coords[0][0], devices_coords[0][1]], [audio_source_coords[0], audio_source_coords[1]]), 3)
        pygame.draw.circle(screen, (250, 0, 50), (devices_coords[1][0], devices_coords[1][1]), find_coord_dist([devices_coords[1][0], devices_coords[1][1]], [audio_source_coords[0], audio_source_coords[1]]), 3)
        pygame.draw.circle(screen, (250, 0, 50), (devices_coords[2][0], devices_coords[2][1]), find_coord_dist([devices_coords[2][0], devices_coords[2][1]], [audio_source_coords[0], audio_source_coords[1]]), 3)
        pygame.draw.circle(screen, (250, 0, 50), (devices_coords[3][0], devices_coords[3][1]), find_coord_dist([devices_coords[3][0], devices_coords[3][1]], [audio_source_coords[0], audio_source_coords[1]]), 3)
 


    pygame.display.flip()

pygame.quit()