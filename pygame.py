import pygame
import random
import math
import requests
import json
import librosa
import asyncio
import aiohttp
import time
import gzip


WINDOW_SIZE = [1600, 1000] #give multiple of 200 

COORD_VARIATION_CONST = [WINDOW_SIZE[0]//22.5, WINDOW_SIZE[1]//22.5]

SPECT_AUDIO_CLASSIFICATION_API = 'http://127.0.0.1:5000/classify-audio-data/spectrogram'
VGGISH_AUDIO_CLASSIFICATION_API = 'http://127.0.0.1:5000/classify-audio-data/vggish'

TDOA_ESTIMATION_API = "http://127.0.0.1:5000/localize-audio-source/tdoa/"

SENSOR_SPRITE_PATH = "pygame_utils/assets/sensor-sprit.png"
SAWING_SPRITE_PATH = "pygame_utils/assets/saw-resized-sprite.png"
AMBIENT_SPRITE_PATH = 'pygame_utils/assets/Ambient-sound-sprite.png'

PYGAME_SAWING_AUDIO_FILEPATH = "pygame_utils/audio_files/background_sawing.wav"
SAWING_AUDIO_FILEPATH = "pygame_utils/audio_files/merged_sawing_ambient.wav"
AMBIENT_AUDIO_FILEPATH = "pygame_utils/audio_files/10s_ambient.wav"

NUM_OF_TREES = 15

#choose which prediction api
prediction_api = SPECT_AUDIO_CLASSIFICATION_API

# Utility functions
def generate_audio_source_coords():
    """Generate audio source coordinates, likely to be around the center of the screen"""
    window_center = [i / 2 for i in WINDOW_SIZE]
    variation_num = [random.randint(-COORD_VARIATION_CONST[0], COORD_VARIATION_CONST[0]),
                     random.randint(-COORD_VARIATION_CONST[1], COORD_VARIATION_CONST[1])]
    return [window_center[0] + variation_num[0], window_center[1] + variation_num[1]]

audio_source_coords = generate_audio_source_coords()


def generate_device_coords(devices_number=4):
    """ come up with an algo to swarm the screen with positions according to number of devices """
    #hard code for 4 devices for now
    window_center = [i/2 for i in WINDOW_SIZE]
    variation_num = [random.randint(- COORD_VARIATION_CONST[0], COORD_VARIATION_CONST[0]), random.randint(- COORD_VARIATION_CONST[1], COORD_VARIATION_CONST[1])]
    variation_num = [i*1.5 for i in variation_num]

    bottom_left = [window_center[0]- (window_center[0]/2) + variation_num[0], window_center[1]- (window_center[1]/2) + variation_num[1]]
    variation_num = [random.randint(- COORD_VARIATION_CONST[0], COORD_VARIATION_CONST[0]), random.randint(- COORD_VARIATION_CONST[1], COORD_VARIATION_CONST[1])]
    variation_num = [i*1.5 for i in variation_num]


    top_left = [window_center[0]- (window_center[0]/2) + variation_num[0], window_center[1]+ (window_center[1]/2) + variation_num[1]]
    variation_num = [random.randint(- COORD_VARIATION_CONST[0], COORD_VARIATION_CONST[0]), random.randint(- COORD_VARIATION_CONST[1], COORD_VARIATION_CONST[1])]
    variation_num = [i*1.5 for i in variation_num]

    bottom_right = [window_center[0]+ (window_center[0]/2) + variation_num[0], window_center[1]- (window_center[1]/2) + variation_num[1]]
    variation_num = [random.randint(- COORD_VARIATION_CONST[0], COORD_VARIATION_CONST[0]), random.randint(- COORD_VARIATION_CONST[1], COORD_VARIATION_CONST[1])]
    variation_num = [i*1.5 for i in variation_num]

    top_right = [window_center[0]+ (window_center[0]/2) + variation_num[0], window_center[1]+ (window_center[1]/2) + variation_num[1]]

    return [bottom_left, top_left, top_right, bottom_right]

devices_coords = generate_device_coords()


def mult_factor(dist):
    """Calculate a number between 0 and 1 inversely proportional to the distance"""
    MAX_DIST_BW_AUDIO_DEVICE = 670
    return 1 - dist / MAX_DIST_BW_AUDIO_DEVICE


def generate_audio_data_for_dist(source_audio, distance):
    """Generate varying intensity of audio data for different distances"""
    return source_audio * mult_factor(distance)


menu_isAmbient = True
menu_isSawing = False

get_audio_source_sprite = lambda: AMBIENT_SPRITE_PATH if menu_isAmbient else SAWING_SPRITE_PATH


class AudioSource:
    """Class representing an audio source"""
    def __init__(self, coords):
        self.coords = coords
        self.is_sawing = False
        self.load_asset()

    def load_asset(self, source_path=get_audio_source_sprite()):
        saw_tile = pygame.image.load(source_path)
        screen.blit(saw_tile, (self.coords[0] - 100, self.coords[1] - 100))

    def create_audio_source_loop(self, source_audio_clips_list, is_sawing, buffer_frame_size=2,
                                 buffer_max_frames=3):
        pass

    def stream_audio_data(self):
        pass

    def relocate_coords(self, coords):
        self.coords = coords


class SensorDevice:
    """Class representing a sensor device"""
    def __init__(self, coords):
        self.coords = coords
        self.is_sawing = False
        self.load_asset()

    def load_asset(self, source_path=SENSOR_SPRITE_PATH):
        pygame.draw.circle(screen, (0, 0, 250), (self.coords[0], self.coords[1]), 25)

    def receive_audio_data(self, original_audio_data, distance_bw_audio_source_sensor):
        return generate_audio_data_for_dist(original_audio_data, distance_bw_audio_source_sensor)

    def receive_streaming_audio(self, received_audio_data):
        pass

    def predict_audio_data(self, received_audio_data):
        r = requests.post(SPECT_AUDIO_CLASSIFICATION_API, data=json.dumps({"audio_data": received_audio_data.tolist()}),
                          headers={"Content-Type": "application/json"})
        return r.json()

    async def async_predict_audio_data(self, session, received_audio_data):
        compressed_audio_data = gzip.compress(received_audio_data.tobytes())
        headers = {'Content-Encoding': 'gzip'}
        async with session.post(prediction_api, data=compressed_audio_data,
                                headers=headers) as response:
            result = await response.json()
            return result

    def relocate_coords(self, coords):
        self.coords = coords


def generate_sensors_data_tdoa(sensors_coords, distances_arr):
    sensors_data = []
    for i, (coords, distance) in enumerate(zip(sensors_coords, distances_arr)):
        sensor_data = {
            f"sensor{i+1}": {
                "coords": coords,
                "distance_to_audio_source": distance
            }
        }
        sensors_data.append(sensor_data)
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

menu_isAmbient = True
menu_isSawing = False

# source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else "audio_files/background_sawing.wav"
source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else PYGAME_SAWING_AUDIO_FILEPATH



pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)

#init background
background = pygame.Surface(WINDOW_SIZE)
background.fill((0,0,0))

#render background
tile_image = pygame.image.load('pygame_utils/assets/grass-resized-sprit.png')
tree_image = pygame.image.load('pygame_utils/assets/tree-resized-sprit.png')

num_tiles_x = WINDOW_SIZE[0] // tile_image.get_width() + 1
num_tiles_y = WINDOW_SIZE[1] // tile_image.get_height() + 1

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


RESET_BUTTON_DIMS= [WINDOW_SIZE[0]/13.5, WINDOW_SIZE[1]/20]
RESET_BUTTON_COORDS = [WINDOW_SIZE[0]/1.1799, WINDOW_SIZE[1]/50]

RUN_BUTTON_DIMS = [WINDOW_SIZE[0]/20, WINDOW_SIZE[1]/20]
RUN_BUTTON_COORDS = [WINDOW_SIZE[0]/1.081, WINDOW_SIZE[1]/50]

DROP_DOWN_BUTTON_DIMS = [WINDOW_SIZE[0]/60,WINDOW_SIZE[1]/20]
DROP_BUTTON_COORDS = [WINDOW_SIZE[0]/1.020, WINDOW_SIZE[1]/50]

DISPLAY_TIME_COORDS = [WINDOW_SIZE[0]/2.15, WINDOW_SIZE[1]/50]


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

play_audio(source_audio_file_path)
playing_audio = source_audio_file_path


#rendering gui components
def handle_quit_event(event):
    if event.type == pygame.QUIT:
        return False
    return True

def handle_reset_button(mouse_pos, coords, dims, sensor_list, num_of_trees, audio_source_coords, devices_coords, rand_tree_coords):
    if coords[0] <= mouse_pos[0] <= coords[0] + dims[0] and coords[1] <= mouse_pos[1] <= coords[1] + dims[1]:
        audio_source_coords = generate_audio_source_coords()
        devices_coords = generate_device_coords()
        rand_tree_coords = handle_tree_generation(num_of_trees)
        for sensor in sensor_list:
            sensor.is_sawing = False
        pygame.display.flip()
    return audio_source_coords, devices_coords, rand_tree_coords

import threading

def handle_run_button_in_thread(mouse_pos, coords, dims, sensor_list, is_tdoa):
    threading.Thread(target=handle_run_button, args=(mouse_pos, coords, dims, sensor_list, is_tdoa)).start()

def handle_run_button(mouse_pos, coords, dims, sensor_list, is_tdoa):
    if coords[0] <= mouse_pos[0] <= coords[0] + dims[0] and coords[1] <= mouse_pos[1] <= coords[1] + dims[1]:
        source_audio_data = ambient_audio_data if menu_isAmbient else sawing_audio_data
        received_audio_data_all_sensors = []
        for idx, sensor in enumerate(sensor_list):
            received_audio_data = sensor.receive_audio_data(source_audio_data, find_coord_dist(sensor.coords,audio_source.coords))
            received_audio_data_all_sensors.append(received_audio_data)
        
        async def gather_pred_data_async():

            async def main():
                async with aiohttp.ClientSession() as session:
                    responses = await asyncio.gather(
                        sensor1.async_predict_audio_data(session, received_audio_data_all_sensors[0]),
                        sensor2.async_predict_audio_data(session, received_audio_data_all_sensors[1]),
                        sensor3.async_predict_audio_data(session, received_audio_data_all_sensors[2]),
                        sensor3.async_predict_audio_data(session, received_audio_data_all_sensors[3])
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
            sensors_data = generate_sensors_data_tdoa(devices_coords,sensor_source_dist_arr) 
            tdoa_est = estimate_tdoa(sensors_data)
            
            print("actual coords: ", audio_source_coords)            
            print("tdoa prediction: ", tdoa_est)       
        
        if prediction_api == SPECT_AUDIO_CLASSIFICATION_API:
            for i, prediction in enumerate(devices_predictions):
                sensor = sensor_list[i]
                sensor.is_sawing = True if 'sawing' in prediction else sensor.is_sawing
        if prediction_api == VGGISH_AUDIO_CLASSIFICATION_API:
            for i, prediction in enumerate(devices_predictions):
                sensor = sensor_list[i]
                sensor.is_sawing = True if prediction == 1 else sensor.is_sawing

        return devices_predictions

def handle_dropdown_button(mouse_pos, coords, dims, menu_isVisible, menu_isSawing, menu_isAmbient, is_tdoa):
    if coords[0] <= mouse_pos[0] <= coords[0] + dims[0] and coords[1] <= mouse_pos[1] <= coords[1] + dims[1]:
        menu_isVisible = not menu_isVisible

    if menu_isVisible:

        if coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and coords[1]+45+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= coords[1]+45+(menu_height)//len(menu_contents) + radio_button_radius +10:
            menu_isSawing = True
            menu_isAmbient = False

        if coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and coords[1]+75+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= coords[1]+75+(menu_height)//len(menu_contents) + radio_button_radius +10:
            menu_isSawing = False
            menu_isAmbient = True

        if coords[0]-50 + menu_width // 2 -10 <= mouse_pos[0] <= coords[0]-50 + menu_width // 2 + radio_button_radius + 10 and coords[1]+105+(menu_height)//len(menu_contents) -10<= mouse_pos[1] <= coords[1]+105+(menu_height)//len(menu_contents) + radio_button_radius +10:
            is_tdoa = not is_tdoa

    return menu_isVisible, menu_isSawing, menu_isAmbient, is_tdoa

def handle_time_display(screen):
    time_string = current_time_string()
    font = pygame.font.Font(None,40)
    text_surface = font.render(time_string, True, (255,255,255))
    screen.blit(text_surface,(DISPLAY_TIME_COORDS[0], DISPLAY_TIME_COORDS[1]))

def render_background(screen, tile_image, num_tiles_x, num_tiles_y):
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            tile_x = x * tile_image.get_width()
            tile_y = y * tile_image.get_height()
            screen.blit(tile_image, (tile_x, tile_y))

def render_trees(screen, tree_coords, tree_image):
    for tree_coord in tree_coords:
        x, y = tree_coord[0], tree_coord[1]
        screen.blit(tree_image, (x, y))

font = pygame.font.Font(None, 36)    

def render_reset_button(screen, coords, dims):
    pygame.draw.rect(screen, (255, 0, 0), (coords[0], coords[1], dims[0], dims[1]))
    font = pygame.font.Font(None, 36)
    reset_text = font.render("Reset", True, (255, 255, 255))
    text_rect = reset_text.get_rect(center=(coords[0] + dims[0] // 2, coords[1] + dims[1] // 2))
    screen.blit(reset_text, text_rect)

def render_run_button(screen, coords, dims):
    pygame.draw.rect(screen, (255, 0, 0), (coords[0], coords[1], dims[0], dims[1]))
    run_text = font.render("Run", True, (255, 255, 255))
    text_rect = run_text.get_rect(center=(coords[0] + dims[0] // 2, coords[1] + dims[1] // 2))
    screen.blit(run_text, text_rect)

def render_dropdown_button(screen, coords, dims, menu_isVisible):
    get_dropdown_symbol = "^" if menu_isVisible else "v"
    pygame.draw.rect(screen, (255, 0, 0), (coords[0], coords[1], dims[0], dims[1]))
    drop_text = font.render(get_dropdown_symbol, True, (255, 255, 255))
    text_rect = drop_text.get_rect(center=(coords[0] + dims[0] // 2, coords[1] + dims[1] // 2))
    screen.blit(drop_text, text_rect)

def render_dropdown_contents(screen, menu_isVisible, menu_isSawing, menu_isAmbient, is_tdoa):
    if menu_isVisible:
        pygame.draw.rect(screen, (255, 255, 255), (DROP_BUTTON_COORDS[0] - 100, DROP_BUTTON_COORDS[1] + 60, menu_width, menu_height))
        for i, item in enumerate(menu_contents):
            text = font.render(item, True, (0, 0, 0))
            text_rect = text.get_rect(center=(DROP_BUTTON_COORDS[0] - 115 + menu_width // 2, DROP_BUTTON_COORDS[1] + 60 + (menu_height // len(menu_contents)) * i + menu_height // (2 * len(menu_contents))))
            screen.blit(text, text_rect)
            
            # radio_button_radius = 5
            if menu_isSawing and i == 0:
                pygame.draw.circle(screen, (0, 0, 0), (DROP_BUTTON_COORDS[0] - 50 + menu_width // 2, DROP_BUTTON_COORDS[1] + 45 + (menu_height) // len(menu_contents)), radio_button_radius)
            if menu_isAmbient and i == 1:
                pygame.draw.circle(screen, (0, 0, 0), (DROP_BUTTON_COORDS[0] - 50 + menu_width // 2, DROP_BUTTON_COORDS[1] + 75 + (menu_height) // len(menu_contents)), radio_button_radius)
            if is_tdoa and i == 2:
                pygame.draw.circle(screen, (0, 0, 0), (DROP_BUTTON_COORDS[0] - 50 + menu_width // 2, DROP_BUTTON_COORDS[1] + 105 + (menu_height) // len(menu_contents)), radio_button_radius)

def render_audio_source(screen, audio_source_coords, audio_source):
    audio_source.relocate_coords(audio_source_coords)
    audio_source.load_asset(get_audio_source_sprite())

def render_devices(screen, devices_coords, sensor_list):
    for i, sensor in enumerate(sensor_list):
        sensor.relocate_coords(devices_coords[i])
        sensor.load_asset()
        if sensor.is_sawing:
            pygame.draw.circle(screen, (250, 0, 0), (devices_coords[i][0], devices_coords[i][1]), 35, 5)

def render_tdoa_circles(screen, devices_coords, audio_source_coords, is_tdoa):
    if is_tdoa:
        for device_coord in devices_coords:
            pygame.draw.circle(screen, (250, 0, 50), (device_coord[0], device_coord[1]), find_coord_dist(device_coord, audio_source_coords), 3)

while running:
    for event in pygame.event.get():
        running = handle_quit_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()

            audio_source_coords, devices_coords, rand_tree_coords = handle_reset_button(mouse_pos, RESET_BUTTON_COORDS, RESET_BUTTON_DIMS, [sensor1, sensor2, sensor3, sensor4], NUM_OF_TREES, audio_source_coords, devices_coords, rand_tree_coords)
            audio_source_predictions = handle_run_button_in_thread(mouse_pos, RUN_BUTTON_COORDS, RUN_BUTTON_DIMS, [sensor1, sensor2, sensor3, sensor4], is_tdoa)
            menu_isVisible, menu_isSawing, menu_isAmbient, is_tdoa = handle_dropdown_button(mouse_pos, DROP_BUTTON_COORDS, DROP_DOWN_BUTTON_DIMS, menu_isVisible, menu_isSawing, menu_isAmbient, is_tdoa)

    screen.blit(background, (0, 0))

    source_audio_file_path = AMBIENT_AUDIO_FILEPATH if menu_isAmbient else PYGAME_SAWING_AUDIO_FILEPATH
    handle_background_audio(playing_audio)
    playing_audio = source_audio_file_path

    render_background(screen, tile_image, num_tiles_x, num_tiles_y)
    render_trees(screen, rand_tree_coords, tree_image)
    render_reset_button(screen, RESET_BUTTON_COORDS, RESET_BUTTON_DIMS)
    render_run_button(screen, RUN_BUTTON_COORDS, RUN_BUTTON_DIMS)
    render_dropdown_button(screen, DROP_BUTTON_COORDS, DROP_DOWN_BUTTON_DIMS, menu_isVisible)
    render_dropdown_contents(screen, menu_isVisible, menu_isSawing, menu_isAmbient, is_tdoa)
    handle_time_display(screen)
    render_audio_source(screen, audio_source_coords, audio_source)
    render_devices(screen, devices_coords, [sensor1, sensor2, sensor3, sensor4])
    render_tdoa_circles(screen, devices_coords, audio_source_coords, is_tdoa)
    pygame.display.flip()

pygame.quit()