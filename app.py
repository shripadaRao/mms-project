#imports
from PIL import Image
import numpy as np
import tensorflow as tf
import librosa
import gzip
from flask import Flask, request, jsonify
import io
import matplotlib
import matplotlib.pyplot as plt
import math
from statistics import mode

import tensorflow as tf
from vggish import vggish_input
from vggish import vggish_slim
# import xgboost as xgb
import joblib

import os
import numpy as np
from pydub import AudioSegment
import soundfile as sf


matplotlib.use('Agg')
plt.switch_backend('Agg')

# ML_MODEL_PATH = "cnn_model_v2.tflite"
ML_MODEL_PATH = "models/cnn_model_v3.tflite"

prediction_classes = {0:'ambient',1:'sawing'}

"""generate 10second spectrogram. slide through 6 iterations of 180 pixels along x axis"""
# def generate_spectrogram_img_memory(audio_data, sr):
#     y = audio_data
#     D = np.abs(librosa.stft(y))**2
#     S = librosa.feature.melspectrogram(S=D, sr=sr)

#     fig, ax = plt.subplots(figsize=(14.5, 6))
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     img = librosa.display.specshow(S_dB, x_axis='time',
#                                    y_axis='mel', sr=sr,
#                                    fmax=6000, ax=ax)

#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     plt.close(fig)

#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     img_arr = np.array(Image.open(buf))
#     # Image.fromarray(img_arr).save('random'+str(np.random.randint(100))+'.png')

#     return img_arr

# # crop_img_dims = [170,200,350,380]       #left, upper, right, lower
# def pre_process_img_v3_data(np_img, dims):
#     left, upper, right, lower = dims[0], dims[1], dims[2], dims[3]
#     img = Image.fromarray(np_img)

#     cropped_img = img.crop((left, upper+50, right, lower+50))

#     grey_img = cropped_img.convert('L')
    
#     grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
#     grey_img = grey_img.point(lambda p: 110 if p < 130 else p)

#     np_img = np.asarray(grey_img)
#     return np_img 

# def predict_with_tflite_data(image_data, model_path):
#     # image = pre_process_img_v3_data(image_data, crop_img_dims)
#     image = image_data
#     image = image / 255.0
#     image = np.expand_dims(image, axis=0)
#     image = np.expand_dims(image, axis=-1)
#     image = image.astype(np.float32)

#     # Load the model
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     # Get the input and output tensors
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Make the prediction
#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     prediction = interpreter.get_tensor(output_details[0]['index'])
#     # print(prediction)
#     # Return the prediction
#     prediction_accuracy = max(prediction[0])
#     class_name = prediction_classes[np.argmax(prediction)]
#     return [class_name, prediction_accuracy]

# def predict_audio_data(audio_data, sr=20400):
#     # print(len(audio_data))
#     # audio_data_string = [i  for i in audio_data]
#     # with open('rand.txt', 'w') as f:
#     #     f.write(str(audio_data_string))
#     spectrogram_img_data = generate_spectrogram_img_memory(audio_data, sr)
#     x = 0
#     pred_dic = {}
#     for i in range(5):
#         crop_img_dims = [180+x,200,360+x,380]
#         processed_img_data = pre_process_img_v3_data(spectrogram_img_data, crop_img_dims)
#         pred = (predict_with_tflite_data(processed_img_data, ML_MODEL_PATH))
#         if pred[0] not in pred_dic:
#             pred_dic[pred[0]] = 1
#         else:
#             pred_dic[pred[0]] +=1
#         x+=180
#     return pred_dic
class SpectAudioPrediction:
    def __init__(self, model_path):
        self.model_path = model_path

    def generate_spectrogram_img_memory(self, audio_data, sr):
        y = audio_data
        D = np.abs(librosa.stft(y)) ** 2
        S = librosa.feature.melspectrogram(S=D, sr=sr)

        fig, ax = plt.subplots(figsize=(14.5, 6))
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=6000, ax=ax)

        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.close(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.array(Image.open(buf))

        return img_arr

    def pre_process_img_v3_data(self, np_img, dims):
        left, upper, right, lower = dims[0], dims[1], dims[2], dims[3]
        img = Image.fromarray(np_img)

        cropped_img = img.crop((left, upper + 50, right, lower + 50))

        grey_img = cropped_img.convert('L')

        grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
        grey_img = grey_img.point(lambda p: 110 if p < 130 else p)

        np_img = np.asarray(grey_img)
        return np_img

    def predict_with_tflite_data(self, image_data):
        image = image_data / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)

        # Load the model
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        # Get the input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make the prediction
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Return the prediction
        prediction_accuracy = max(prediction[0])
        class_name = prediction_classes[np.argmax(prediction)]
        return [class_name, prediction_accuracy]

    def predict_audio_data(self, audio_data, sr=20400):
        spectrogram_img_data = self.generate_spectrogram_img_memory(audio_data, sr)
        x = 0
        pred_dic = {}
        for i in range(5):
            crop_img_dims = [180 + x, 200, 360 + x, 380]
            processed_img_data = self.pre_process_img_v3_data(spectrogram_img_data, crop_img_dims)
            pred = self.predict_with_tflite_data(processed_img_data)
            if pred[0] not in pred_dic:
                pred_dic[pred[0]] = 1
            else:
                pred_dic[pred[0]] += 1
            x += 180
        return pred_dic

class VGGishAudioClassifier:
    XGB_MODEL_PATH = "models/audio_classification_model.xgb"
    RF_MODEL_PATH = "models/random_forest_optimized.pkl"

    def __init__(self, wav_data):
        self.load_ml_classifier()
        self.wav_data = wav_data

    def pre_process_audio_data(self):
        # wav_data_processed = np.mean(self.wav_data, axis=1)
        wav_data_processed = self.wav_data
        wav_data_processed = librosa.resample(y=wav_data_processed,orig_sr=22050,target_sr= 16000)
        return wav_data_processed

    def extract_audio_features(self):
        features = []
        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish/vggish_model.ckpt')
            tf_features_tensor = sess.graph.get_tensor_by_name('vggish/input_features:0')
            tf_embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')

            wav_data = vggish_input.waveform_to_examples(self.pre_process_audio_data(), sample_rate=16000)
            features_batch = sess.run(tf_embedding_tensor, feed_dict={tf_features_tensor: wav_data})
            features.append(features_batch)

        return np.vstack(features)
    
    def load_ml_classifier(self):
        # self.xgb_classifier = xgb.XGBClassifier().load_model(self.XGB_MODEL_PATH)
        self.rf_classifier = joblib.load(self.RF_MODEL_PATH)

    def predict(self):
        features = self.extract_audio_features()
        # print("features: ", features)
        prediction = self.rf_classifier.predict(features)[0]
        return (prediction)


"""
    TDOA - sensors data object received is of below data model
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
    

def get_intersection_points(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return [[x3,y3],[x4,y4]]
    
   
def estimate_tdoa(sensors_data):
    all_sensor_coords_arr = []#[[],[],[],[]]
    all_distances_arr = []

    for sensor_data in sensors_data:
        for sensor_name, sensor_info in sensor_data.items():
            coords = sensor_info["coords"]
            distance = sensor_info["distance_to_audio_source"]
            all_sensor_coords_arr.append(coords)
            all_distances_arr.append(distance)

    all_circles_intersection_coords = []

    for i in range(len(all_sensor_coords_arr)-1):
        for j in range(i+2, len(all_sensor_coords_arr)):
            x1, y1, r1 = all_sensor_coords_arr[i][0], all_sensor_coords_arr[i][1], all_distances_arr[i]
            x2, y2, r2 = all_sensor_coords_arr[j][0], all_sensor_coords_arr[j][1], all_distances_arr[j]
            
            intersection_coords = get_intersection_points(x1, y1, r1, x2, y2, r2)
            if intersection_coords:
                all_circles_intersection_coords.append(intersection_coords[0])
                all_circles_intersection_coords.append(intersection_coords[1])

    if all_circles_intersection_coords:
        #round off the coordinates and find the most occuring coordinates
        x_coords, y_coords = [round(intersection_arr[0],2) for intersection_arr in all_circles_intersection_coords], [round(intersection_arr[1],2) for intersection_arr in all_circles_intersection_coords]
        x_coord, y_coord = mode(x_coords), mode(y_coords)
        return [x_coord, y_coord]
    else:
        return None



app = Flask(__name__)

@app.route('/')
def greet():
    return "hello world!"

@app.route('/classify-audio-data/spectrogram', methods=['POST'])
def classify_audio_data():
    # Check if the request contains gzipped data
    if request.headers.get('Content-Encoding') == 'gzip':
        gzipped_data = request.get_data()
        audio_data = np.frombuffer(gzip.decompress(gzipped_data), dtype=np.float32)
    else:
        audio_data_json = request.get_json()
        audio_data = np.array(audio_data_json['audio_data'])

    # Perform audio data classification
    # result = predict_audio_data(audio_data, sr=20400)
    audio_prediction = SpectAudioPrediction(ML_MODEL_PATH)
    result = audio_prediction.predict_audio_data(audio_data, sr=20400)
    print(result)
    return jsonify(result)

@app.route('/classify-audio-data/vggish', methods=['POST'])
def classify_audio_data_vggish():
    if request.headers.get('Content-Encoding') == 'gzip':
        gzipped_data = request.get_data()
        audio_data = np.frombuffer(gzip.decompress(gzipped_data), dtype=np.float32)
    else:
        audio_data_json = request.get_json()
        audio_data = np.array(audio_data_json['audio_data'])

    prediction_model = VGGishAudioClassifier(audio_data)
    result = prediction_model.predict()
    print((result))
    return jsonify(result)

@app.route('/localize-audio-source/tdoa/', methods=['POST'])
def tdoa():
    sensors_data = request.get_json()
    predicted_tdoa_coords = estimate_tdoa(sensors_data)
    print("predicted coords: ",predicted_tdoa_coords)
    return predicted_tdoa_coords



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)
