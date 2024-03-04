# import wave

# # Open the short WAV file
# short_wav = wave.open("dataset/sawing/file1008.wav", "rb")

# # Get the parameters of the short WAV file
# short_params = short_wav.getparams()

# short_duration = short_params.nframes / short_params.framerate

# # Calculate the number of repetitions needed to reach the desired duration
# desired_duration = 10  # in seconds

# repetitions = int(desired_duration / short_duration)

# # Create a new WAV file with the desired duration
# merged_wav = wave.open("10s_sawing.wav", "wb")

# # Set the parameters of the merged WAV file
# merged_wav.setparams(short_params)

# # Repeat the short WAV file and write it to the merged WAV file
# for _ in range(repetitions):
#     short_wav.rewind()
#     merged_wav.writeframes(short_wav.readframes(short_params.nframes))

# # Close the WAV files
# short_wav.close()
# merged_wav.close()
from pydub import AudioSegment

def merge_audio_clips(sawing_file_path, ambient_file_path):
    # Load the audio files
    sawing_audio = AudioSegment.from_file(sawing_file_path)
    ambient_audio = AudioSegment.from_file(ambient_file_path)

    # Adjust the duration of the audio clips
    sawing_audio_ = sawing_audio[:2000] - 20
    ambient_audio_ = ambient_audio[:4000] + 11

    # Merge the audio clips by overlaying them
    merged_audio1 = ambient_audio_.overlay(sawing_audio_)

    sawing_audio = sawing_audio[-2000:] - 25
    ambient_audio = ambient_audio[-2000:] + 12
    merged_audio2 = ambient_audio.overlay(sawing_audio)

    total_audio_data = merged_audio1 + ambient_audio[2000:-2000] + merged_audio2

    # Export the merged audio as a new file
    merged_file_path = "audio_files/background_sawing.wav"
    total_audio_data.export(merged_file_path, format="wav")

    return merged_file_path

# Example usage
sawing_file_path = "audio_files/10s_sawing.wav"
ambient_file_path = "audio_files/10s_ambient.wav"

merged_file_path = merge_audio_clips(sawing_file_path, ambient_file_path)
print("Merged audio file:", merged_file_path)
