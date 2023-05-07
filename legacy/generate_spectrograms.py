import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa.display

out_dir1 = "dataset/spectrograms/ambient/"
out_dir2 = "dataset/spectrograms/sawing/"

in_dir1 = "dataset/ambient/"
in_dir2 = "dataset/sawing/"

def generateSpectrogram(audioFileName, in_dir, out_dir):
    print("AudioFileName: ",audioFileName)
    y, sr = librosa.load(in_dir + audioFileName, sr=20400)

    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.savefig(out_dir + audioFileName.split('.')[0] + '.png')
    plt.close(fig)


# for fileName in os.listdir(in_dir1):
#   # print(fileName)
#   generateSpectrogram( fileName, in_dir1, out_dir1 )

# print('done with ambient!')


# for fileName in os.listdir(in_dir2):
#   # print(fileName)
#   generateSpectrogram( fileName, in_dir2, out_dir2 )


#rewriting with multiprocessing
import multiprocessing as mp
num_processes = mp.cpu_count()
pool = mp.Pool(num_processes)

def generate_spectrograms_multiprocessing(in_dir, out_dir):
    results = []
    for f in os.listdir(out_dir):
        result = pool.apply_async(generateSpectrogram, args=(f, in_dir, out_dir))
        results.append(result)

    # Wait for all processes to complete
    for result in results:
        result.wait()

    # Close the pool to free up resources
    pool.close()
    pool.join()

generate_spectrograms_multiprocessing(in_dir2, out_dir2)

if __name__ == '__main__':

    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)

    files = os.listdir(in_dir2)

    # Create a pool of processes
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)

    # Use the pool to process the files
    results = []
    for f in files:
        result = pool.apply_async(generateSpectrogram, args=(f, in_dir2, out_dir2))
        results.append(result)

    # Wait for all processes to complete
    for result in results:
        result.wait()

    # Close the pool to free up resources
    pool.close()
    pool.join()