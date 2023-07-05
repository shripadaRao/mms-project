from pytube import YouTube
import time
from pydub import AudioSegment
import pandas as pd
import csv
import uuid
import numpy as np
from multiprocessing import Pool
import os
from pytube.exceptions import VideoPrivate
import urllib.error
import pytube
import json
import sqlite3
import shutil
import librosa



DB_PATH = "yt_dataset/metadata/positive_tag_db/"



def download_yt_audio(yt_url, out_path):
    yt = YouTube(yt_url)
    t = yt.streams.filter(only_audio=True).all()
    t[0].download(filename=out_path)


def duplicate_file(original_outpath):
    duplicate_file_path = original_outpath + "_duplicate"
    shutil.copyfile(original_outpath, duplicate_file_path)

def inject_noise(audio_path, noise_level=0.1):
    audio, sr = librosa.load(audio_path, sr=None)
    noise = np.random.normal(0, noise_level, len(audio))
    noisy_audio = audio + noise
    noisy_audio = librosa.util.normalize(noisy_audio)
    return noisy_audio, sr

def decode_tag(tag):
    decoded_tag = '/' + tag[0] + '/' + tag[1:]
    return decoded_tag

def yt_url_from_id(yt_id):
    # return "https://www.youtu.be/" + yt_id 
    return "https://www.youtube.com/watch?v=" + yt_id

def extract_audio_segment(input_file, output_file, start_time, end_time):
    audio = AudioSegment.from_file(input_file)
    segment = audio[start_time * 1000 : end_time * 1000]  # Convert time to milliseconds
    segment.export(output_file, format="wav")

def read_csv(filepath):
    return pd.read_csv(filepath)

def get_time_stamps(df, yt_id):
    req_record = df[df["yt_id"]] == yt_id
    start_sec = req_record['start_seconds'].values[0]
    end_sec = req_record['end_seconds'].values[0]
    return [start_sec, end_sec]


dataset_path = "yt_dataset/metadata/positive-metadata.csv"


def handle_downloads(yt_url, audio_file_path):
    start_time = time.time()
    download_yt_audio(yt_url, audio_file_path)

def writeInJsonFile(data):
    with open('positive_audio_tag.json', 'w') as f:

        json.dump(data, f)

# positive_tag_dic = {
#         "m01b82r": [],
#         "m01j4z9": [],
#         "m0_ksk": []
#       }

# def jsonTagTitleData(tags_field,title, tag_dict, lock):
#     with lock:
#         for tag in tags_field:
#             if tag in tag_dict:
#                 tag_dict[tag].append(title)

def write_to_database(table_name, yt_id, file_title):
    conn = sqlite3.connect(DB_PATH + 'positive_tags.db')
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {table_name} (file_title, yt_id) VALUES (?, ?)", (file_title, yt_id))
    conn.commit()
    conn.close()

def handle_db_entries(tags_string,yt_id, file_title):
    #hardcode
    if "m01b82r" in tags_string:
        write_to_database("m01b82r",yt_id, file_title)
    
    if "m01j4z9" in tags_string:
        write_to_database("m01j4z9",yt_id, file_title)

    if "m0_ksk" in tags_string:
        write_to_database("m0_ksk",yt_id, file_title)  
    return 


def process_row(index):
    row = df.loc[index]
    
    yt_url = yt_url_from_id(row['yt_id'])
    tag_field = row['tags_modified']

    print(yt_url)
    uuid_string = str(uuid.uuid4())
    audio_file_path = "yt_dataset/yt_audio/positive/"+ uuid_string + '.mp4'
    
    #cant be private, unavailable, cant take much time
    yt = YouTube(yt_url)
    try:
        yt.age_restricted
        try:
            download_yt_audio(yt_url,audio_file_path)
        except VideoPrivate:
            pass
    except:
        pass


    #extract the necessary part
    if os.path.exists(audio_file_path):
        extract_audio_segment(audio_file_path,audio_file_path, row['start_seconds'], row['end_seconds'])

        # add duplicates
        #write to resp table in db
        handle_db_entries(tag_field, row['yt_id'],uuid_string)

    return

if __name__ == '__main__':
    num_processes = 6  # Number of parallel processes

    df = read_csv(dataset_path)
    # df = df[5145:]
    indices = list(df.index)

    # Create a multiprocessing pool
    pool = Pool(processes=num_processes)

    # Apply the process_row function to each index in parallel
    pool.map(process_row, indices)

    # Close the multiprocessing pool
    pool.close()
    pool.join()