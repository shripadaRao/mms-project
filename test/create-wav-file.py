import wave

# Open the short WAV file
short_wav = wave.open("dataset/sawing/file1008.wav", "rb")

# Get the parameters of the short WAV file
short_params = short_wav.getparams()

short_duration = short_params.nframes / short_params.framerate

# Calculate the number of repetitions needed to reach the desired duration
desired_duration = 10  # in seconds

repetitions = int(desired_duration / short_duration)

# Create a new WAV file with the desired duration
merged_wav = wave.open("10s_sawing.wav", "wb")

# Set the parameters of the merged WAV file
merged_wav.setparams(short_params)

# Repeat the short WAV file and write it to the merged WAV file
for _ in range(repetitions):
    short_wav.rewind()
    merged_wav.writeframes(short_wav.readframes(short_params.nframes))

# Close the WAV files
short_wav.close()
merged_wav.close()