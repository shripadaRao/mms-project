import math
import numpy as np
import tensorflow as tf
import librosa
import gzip
from flask import Flask, request, jsonify
from io import BytesIO
import matplotlib.pyplot as plt
from statistics import mode
from vggish import vggish_input, vggish_slim
from PIL import Image
from joblib import load

# ML model paths
ML_MODEL_PATH = "models/cnn_model_v3.tflite"
VGGISH_MODEL_PATH = "vggish/vggish_model.ckpt"
XGB_MODEL_PATH = "models/audio_classification_model.xgb"  # Not used in this version
RF_MODEL_PATH = "models/random_forest_optimized.pkl"

# Prediction classes
prediction_classes = {"ambient": 0, "sawing": 1}


class SpectrogramAudioPrediction:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def generate_spectrogram(self, audio_data, sr):
        y = audio_data
        S = librosa.feature.melspectrogram(S=np.abs(librosa.stft(y)) ** 2, sr=sr)
        fig, ax = plt.subplots(figsize=(14.5, 6))
        librosa.display.specshow(
            librosa.power_to_db(S, ref=np.max), x_axis="time", y_axis="mel", sr=sr, fmax=6000, ax=ax
        )
        plt.close(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_arr = np.array(Image.open(buf))
        return img_arr

    def pre_process_img(self, img_data, crop_dims):
        img = Image.fromarray(img_data)
        cropped_img = img.crop(crop_dims)
        grey_img = cropped_img.convert("L")
        grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
        grey_img = grey_img.point(lambda p: 110 if p < 130 else p)
        return np.asarray(grey_img)

    def predict(self, image_data):
        image = image_data / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]["index"], image)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(output_details[0]["index"])

        prediction_accuracy = np.max(prediction[0])
        class_name = prediction_classes[np.argmax(prediction)]
        return class_name, prediction_accuracy

    def predict_audio(self, audio_data, sr=20400):
        spectrogram_img = self.generate_spectrogram(audio_data, sr)
        predictions = {}
        for i in range(5):
            crop_dims = [180 * i, 200, 360 * i + 180, 380]
            processed_img = self.pre_process_img(spectrogram_img, crop_dims)
            class_name, _ = self.predict(processed_img)
            predictions[class_name] = predictions.get(class_name, 0) + 1
        return max(predictions, key=predictions.get)


class VGGishAudioClassifier:
    def __init__(self, audio_data):
        self.audio_data = audio_data
        self.rf_classifier = load(RF_MODEL_PATH)

    def pre_process_audio(self):
        return librosa.resample(y=self.audio_data, orig_sr=22050, target_sr=16000)

    def extract_features(self):
        features = []
        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_MODEL_PATH)
            tf_features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
            tf_embedding_tensor = sess.graph.get_tensor_by_name("vggish/embedding:0")

            processed_audio = self.pre_process_audio()
            features_batch = sess.run(
                tf_embedding_tensor, feed_dict={tf_features_tensor: vggish_input.waveform_to_examples(processed_audio, sample_rate=16000)}
            )
            features.append(features_batch)
        return np.vstack(features)

    def predict(self):
        features = self.extract_features()
        prediction = self.rf_classifier.predict(features)[0]
        return prediction


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
        for j in range(i+2, len(all_sensor_coords_arr)):
            x1, y1, r1 = all_sensor_coords_arr[i][0], all_sensor_coords_arr[i][1], all_distances_arr[i]
            x2, y2, r2 = all_sensor_coords_arr[j][0], all_sensor_coords_arr[j][1], all_distances_arr[j]
            
            intersection_coords = get_intersection_points(x1, y1, r1, x2, y2, r2)
            if intersection_coords:
                all_circles_intersection_coords.append(intersection_coords[0])
                all_circles_intersection_coords.append(intersection_coords[1])

    if all_circles_intersection_coords:
        x_coords, y_coords = [round(intersection_arr[0],2) for intersection_arr in all_circles_intersection_coords], [round(intersection_arr[1],2) for intersection_arr in all_circles_intersection_coords]
        x_coord, y_coord = mode(x_coords), mode(y_coords)
        return [x_coord, y_coord]
    else:
        return None


if __name__ == "__main__":
    app = Flask(__name__)

    @app.route("/")
    def greet():
        return "Hello world!"

    @app.route("/classify-audio-data/spectrogram", methods=["POST"])
    def classify_audio_data():
        if request.headers.get("Content-Encoding") == "gzip":
            gzipped_data = request.get_data()
            audio_data = np.frombuffer(gzip.decompress(gzipped_data), dtype=np.float32)
        else:
            audio_data_json = request.get_json()
            audio_data = np.array(audio_data_json["audio_data"])

        audio_prediction = SpectrogramAudioPrediction(ML_MODEL_PATH)
        result = audio_prediction.predict_audio(audio_data)
        return jsonify(result)

    @app.route("/classify-audio-data/vggish", methods=["POST"])
    def classify_audio_data_vggish():
        if request.headers.get("Content-Encoding") == "gzip":
            gzipped_data = request.get_data()
            audio_data = np.frombuffer(gzip.decompress(gzipped_data), dtype=np.float32)
        else:
            audio_data_json = request.get_json()
            audio_data = np.array(audio_data_json["audio_data"])

        prediction_model = VGGishAudioClassifier(audio_data)
        result = prediction_model.predict()
        return jsonify(result)

    @app.route("/localize-audio-source/tdoa/", methods=["POST"])
    def tdoa():
        sensors_data = request.get_json()
        predicted_tdoa_coords = estimate_tdoa(sensors_data)
        return predicted_tdoa_coords

    app.run(debug=True, host="0.0.0.0", port=5000)