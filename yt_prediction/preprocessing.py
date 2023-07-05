
# code to trim and export negative class dataset

import os
import random
from multiprocessing import Pool
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import os
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_slim
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def convert_audio_to_mono(audio):
    return audio.set_channels(1)


def resample_audio(audio, sample_rate):
    return audio.set_frame_rate(sample_rate)


def export_audio_clip(clip, output_path, sample_rate=16000):
    clip_data = np.array(clip.get_array_of_samples(), dtype=np.float32) / 32768.0  # Convert to float32 sample
    sf.write(output_path, clip_data, sample_rate, format='WAV', subtype='FLOAT')


def process_audio_file(audio_file, source_folder, output_folder, clip_duration=5000):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(os.path.join(source_folder, audio_file))

        # Skip processing if the audio file is longer than 350 seconds
        duration = len(audio)
        if duration > 350000:
            return

        # Convert audio to mono channel
        audio = convert_audio_to_mono(audio)

        # Resample audio to 16 kHz
        audio = resample_audio(audio, 16000)

        # Extract clips from the audio
        start_time = 0
        end_time = clip_duration
        while end_time <= duration:
            clip = audio[start_time:end_time]

            # Export the clip as mono 16 kHz float32 sample in the range of [-1.0, +1.0]
            output_file = f"{os.path.basename(audio_file)}_{random.randint(0, 100)}.wav"
            output_path = os.path.join(output_folder, output_file)
            export_audio_clip(clip, output_path)

            start_time += clip_duration
            end_time += clip_duration

    except Exception as e:
        print(f"Error processing audio file: {audio_file}")
        print(f"Error message: {str(e)}")

# checkout ML AAT attempt 3 colab file
source_class_folder = ["source_negative_audio_files/yt_dataset/negative/", "source_positive_audio_files/chainsaw/", "source_positive_audio_files/sawing/"]
output_class_folder = ["processed_audio_files/negative/", "processed_audio_files/positive/chainsaw", "processed_audio_files/negative/sawing"]


def pre_process_audio_files():

    for class_index in range(len(source_class_folder)):
        audio_files = os.listdir(source_class_folder[class_index])[:800]
        num_processes = 4
        pool = Pool(processes=num_processes)
        pool.starmap(process_audio_file, [(file, source_class_folder[class_index], output_class_folder[class_index]) for file in audio_files])

        # Close the pool to release resources
        pool.close()
        pool.join()

## VGGish

#check out the above mentioned colab file for colab setup

def extract_audio_features(audio_files, sess, features_tensor, embeddings):
    features = []
    for file in audio_files:
        wav_data = vggish_input.wavfile_to_examples(file)
        features_batch = sess.run(embeddings, feed_dict={features_tensor: wav_data})
        features.append(features_batch)
    return np.vstack(features)

# Function to load and initialize the VGGish model checkpoint
def load_vggish_model(checkpoint_path):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        with tf.compat.v1.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
            features_tensor = tf_graph.get_tensor_by_name('vggish/input_features:0')
            embeddings = tf_graph.get_tensor_by_name('vggish/embedding:0')
            return sess, features_tensor, embeddings

# Function to list WAV files in a folder
def list_wav_files(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]

# Function to sample random files from a list
def sample_files(file_list, num_samples):
    return random.sample(file_list, num_samples)

# Function to save features to a file
def save_features(features, file_path):
    np.save(file_path, features)


def vggish_main():
    # Set the path to the VGGish checkpoint file
    checkpoint_path = 'vggish_model.ckpt'

    # Load and initialize the VGGish model checkpoint
    sess, features_tensor, embeddings = load_vggish_model(checkpoint_path)

    # Set the paths to the folders containing WAV files
    positive_folder = '/content/processed_audio_files/positive/chainsaw'
    negative_folder = '/content/processed_audio_files/negative'

    # List all WAV files in the positive folder
    positive_files = list_wav_files(positive_folder)

    # List all WAV files in the negative folder and sample 1981 random files
    negative_files = list_wav_files(negative_folder)
    negative_files = sample_files(negative_files, 1981)  ## 2000 was used in colab

    # Extract VGGish features from audio files
    positive_features = extract_audio_features(positive_files, sess, features_tensor, embeddings)
    save_features(positive_features, 'positive_features.npy')

    negative_features = extract_audio_features(negative_files, sess, features_tensor, embeddings)
    save_features(negative_features, 'negative_features.npy')



