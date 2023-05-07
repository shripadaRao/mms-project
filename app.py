#imports
from PIL import Image
import numpy as np
import time
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import librosa.display
import io

ML_MODEL_PATH = "cnn_model_v2.tflite"

prediction_classes = {0:'ambient',1:'sawing'}

def generate_spectrogram_img_memory(audio_file):

    y, sr = librosa.load(audio_file, sr=20400)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.close(fig)


    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.array(Image.open(buf))

    return img_arr


def pre_process_img_v3(source_filepath):
    left, upper, right, lower = 170, 200, 350, 380
    with Image.open(source_filepath) as img:
        cropped_img = img.crop((left, upper, right, lower))

    grey_img = cropped_img.convert('L')
    
    grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
    grey_img = grey_img.point(lambda p: 110 if p < 130 else p)

    np_img = np.asarray(grey_img)
    return np_img

def pre_process_img_v3_data(np_img):
    left, upper, right, lower = 170, 200, 350, 380
    img = Image.fromarray(np_img)

    cropped_img = img.crop((left, upper, right, lower))

    grey_img = cropped_img.convert('L')
    
    grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
    grey_img = grey_img.point(lambda p: 110 if p < 130 else p)

    np_img = np.asarray(grey_img)
    return np_img 


def predict_with_tflite_data(image_data, model_path):
    image = pre_process_img_v3_data(image_data)
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

    # Return the prediction
    prediction_accuracy = max(prediction[0])
    class_name = prediction_classes[np.argmax(prediction)]

    return [class_name, prediction_accuracy]

def predict_with_tflite(image_path, model_path):
    image = pre_process_img_v3(image_path,)
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

    # Return the prediction
    prediction_accuracy = max(prediction[0])
    class_name = prediction_classes[np.argmax(prediction)]

    return [class_name, prediction_accuracy]

def predict_audio_file(audio_filepath):
    start = time.time()
    spectrogram_img_data = generate_spectrogram_img_memory(audio_filepath)
    print('spectrogram time: ', time.time() - start)
    class_name, prediction_accuracy = predict_with_tflite_data(spectrogram_img_data, ML_MODEL_PATH)
    print(class_name, prediction_accuracy)
    print('total time: ', time.time()- start)


predict_audio_file('dataset/sawing/file677.wav')
# print(predict_with_tflite('dataset/spectrograms/ambient/file12.png', ML_MODEL_PATH))
