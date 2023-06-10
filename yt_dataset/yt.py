from pytube import YouTube
import time
from pydub import AudioSegment
import pandas as pd
import csv
import uuid
import numpy as np
from multiprocessing import Pool


def download_yt_audio(yt_url, out_path):
    yt = YouTube(yt_url)
    t = yt.streams.filter(only_audio=True).all()
    t[0].download(filename=out_path)

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


# yt_url = yt_url_from_id("--eWXk8if7g")
# print(yt_url)
# download_yt_audio(yt_url,"yt_dataset/yt_audio/test3.mp4")

# start_time = time.time()
# download_yt_audio("https://www.youtube.com/watch?v=xvFZjo5PgG0","yt_dataset/yt_audio/test2.mp4")
# print('download time: ', time.time() - start_time)

# start_time = time.time()
# extract_audio_segment('yt_dataset/yt_audio/test2.mp4',"yt_dataset/yt_audio/test2.mp4",1,4)
# print('extract time: ', time.time() - start_time)

dataset_path = "yt_dataset/main/negative-metadata.csv"

# if __name__ == "__main__":
#     df = read_csv(dataset_path)

#     for index, row in df.iterrows():
#         yt_url = yt_url_from_id(row['yt_id'])
#         print(yt_url)
#         #download yt audio
#         audio_file_path = "yt_dataset/yt_audio/negative/"+ str(index) + '.mp4'
#         download_yt_audio(yt_url,audio_file_path)
#         #extract the necessary part
#         extract_audio_segment(audio_file_path,audio_file_path, row['start_seconds'], row['end_seconds'])

#         break

def process_row(index):
    row = df.loc[index]
    
    yt_url = yt_url_from_id(row['yt_id'])
    print(yt_url)
    #download yt audio
    audio_file_path = "yt_dataset/yt_audio/negative/"+ str(uuid.uuid4()) + '.mp4'
    download_yt_audio(yt_url,audio_file_path)
    #extract the necessary part
    extract_audio_segment(audio_file_path,audio_file_path, row['start_seconds'], row['end_seconds'])



num_processes = 8  # Number of parallel processes

df = read_csv(dataset_path)

indices = list(df.index)

# Create a multiprocessing pool
pool = Pool(processes=num_processes)

# Apply the process_row function to each index in parallel
pool.map(process_row, indices)

# Close the multiprocessing pool
pool.close()
pool.join()