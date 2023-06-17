#imports
from PIL import Image
import numpy as np
import tensorflow as tf
import librosa
import gzip
from flask import Flask, request
import io
import matplotlib
import matplotlib.pyplot as plt
import math
import random


matplotlib.use('Agg')
plt.switch_backend('Agg')

# ML_MODEL_PATH = "cnn_model_v2.tflite"
ML_MODEL_PATH = "models/cnn_model_v3.tflite"

prediction_classes = {0:'ambient',1:'sawing'}

"""generate 10second spectrogram. slide through 6 iterations of 180 pixels along x axis"""
def generate_spectrogram_img_memory(audio_data, sr):
    y = audio_data
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)

    fig, ax = plt.subplots(figsize=(14.5, 6))
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=6000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.close(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    # Image.fromarray(img_arr).save('random'+str(np.random.randint(100))+'.png')

    return img_arr

# crop_img_dims = [170,200,350,380]       #left, upper, right, lower
def pre_process_img_v3_data(np_img, dims):
    left, upper, right, lower = dims[0], dims[1], dims[2], dims[3]
    img = Image.fromarray(np_img)

    cropped_img = img.crop((left, upper+50, right, lower+50))

    grey_img = cropped_img.convert('L')
    
    grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
    grey_img = grey_img.point(lambda p: 110 if p < 130 else p)

    np_img = np.asarray(grey_img)
    return np_img 

def predict_with_tflite_data(image_data, model_path):
    # image = pre_process_img_v3_data(image_data, crop_img_dims)
    image = image_data
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image.astype(np.float32)

    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make the prediction
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    print(prediction)
    # Return the prediction
    prediction_accuracy = max(prediction[0])
    class_name = prediction_classes[np.argmax(prediction)]
    return [class_name, prediction_accuracy]

def predict_audio_data(audio_data, sr=20400):
    # print(len(audio_data))
    # audio_data_string = [i  for i in audio_data]
    # with open('rand.txt', 'w') as f:
    #     f.write(str(audio_data_string))
    spectrogram_img_data = generate_spectrogram_img_memory(audio_data, sr)
    x = 0
    pred_dic = {}
    for i in range(5):
        crop_img_dims = [180+x,200,360+x,380]
        processed_img_data = pre_process_img_v3_data(spectrogram_img_data, crop_img_dims)
        pred = (predict_with_tflite_data(processed_img_data, ML_MODEL_PATH))
        if pred[0] not in pred_dic:
            pred_dic[pred[0]] = 1
        else:
            pred_dic[pred[0]] +=1
        x+=180
    return pred_dic

def find_circle_intersection(x1, y1, r1, x2, y2, r2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if math.isclose(distance, r1 + r2):
        intersection_x = x1 + (r1 * (x2 - x1)) / distance
        intersection_y = y1 + (r1 * (y2 - y1)) / distance
        return [intersection_x, intersection_y]
   
"""
    data object looks like below
[
	{
	  "sensor1" : {
		"coords" : [],
		"distance_to_audio_source" : int
		}
	},
	{
	  "sensor2" : {
		"coords" : [],
		"distance_to_audio_source" : int
		}
	},
	.
	.	
]
	"""
    

def find_avg_intersection(sensors_data):
    all_sensor_coords_arr = []
    all_distances_arr = []

    for sensor_data in sensors_data:
        for sensor_name, sensor_info in sensor_data.items():
            coords = sensor_info["coords"]
            distance = sensor_info["distance_to_audio_source"]
            all_sensor_coords_arr.append(coords)
            all_distances_arr.append(distance)

    all_circles_intersection_coords = []

    for i in range(len(all_sensor_coords_arr)-1):
        for j in range(i+1, len(all_sensor_coords_arr)):
            x1, y1, r1 = all_sensor_coords_arr[i][0], all_sensor_coords_arr[i][1], all_distances_arr[i]
            x2, y2, r2 = all_sensor_coords_arr[j][0], all_sensor_coords_arr[j][1], all_distances_arr[j]
            
            intersection_coords = find_circle_intersection(x1, y1, r1, x2, y2, r2)
            print("intersection_coords: ", intersection_coords)
            if intersection_coords:
                all_circles_intersection_coords.append(intersection_coords)
    print("all intersection: ",all_circles_intersection_coords)
    if all_circles_intersection_coords:
        avg_x = sum(coord[0] for coord in all_circles_intersection_coords) / len(all_circles_intersection_coords)
        avg_y = sum(coord[1] for coord in all_circles_intersection_coords) / len(all_circles_intersection_coords)
        return [avg_x, avg_y]
    else:
        return None



app = Flask(__name__)

@app.route('/')
def greet():
    return "hello world!"

@app.route('/classify-audio-data/', methods=['POST'])
def classify_audio_data():
    # Check if the request contains gzipped data
    if request.headers.get('Content-Encoding') == 'gzip':
        gzipped_data = request.get_data()
        audio_data = np.frombuffer(gzip.decompress(gzipped_data), dtype=np.float32)
    else:
        audio_data_json = request.get_json()
        audio_data = np.array(audio_data_json['audio_data'])

    # Perform audio data classification
    result = predict_audio_data(audio_data, sr=20400)

    return (result)

@app.route('/localize-audio-source/tdoa/', methods=['POST'])
def tdoa():
    sensors_data = request.get_json()
    avg_intersection_coords = find_avg_intersection(sensors_data)
    print(avg_intersection_coords)
    return avg_intersection_coords



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)

# sensor_data = {
#     "sensor1": {
#         "coords": [0, 0],
#         "distance_to_audio_source": random.randint(1, 10)
#     },
#     "sensor2": {
#         "coords": [3, 4],
#         "distance_to_audio_source": random.randint(1, 10)
#     },
#     "sensor3": {
#         "coords": [-2, 2],
#         "distance_to_audio_source": random.randint(1, 10)
#     },
#     # Add more sensor data as needed
# }

# # Test the function
# intersection_point = find_avg_intersection(sensor_data)
# print("Average Intersection Point:", intersection_point)