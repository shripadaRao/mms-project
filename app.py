#imports
from PIL import Image
from keras.models import load_model
import numpy as np
import time

ML_MODEL_PATH = "dataset/cnn_model_v1.h5"

def pre_process_img(source_filepath):
  start = time.time()

  left, upper, right, lower = 80, 60, 475, 425
  with Image.open(source_filepath) as img:
      cropped_img = img.crop((left, upper, right, lower))  
  grey_img = cropped_img.convert('L')
  grey_img = grey_img.point(lambda p: 255 if p > 165 else p)
  grey_img = grey_img.point(lambda p: 110 if p < 130  else p)

  print('pre-processing time: ', time.time() - start)
  return np.asarray(grey_img)

def predict_spectrogram(filepath, model):

#   model = load_model(ML_MODEL_PATH)
  
  img_data = pre_process_img(filepath)
  x = img_data / 255.
  x = np.expand_dims(x, axis=0)
  preds = model.predict(x)
  return True if preds.argmax()==1 else False

while(True):
   model = load_model(ML_MODEL_PATH)

   in_dir1 = "dataset/spectrograms/ambient/"
   in_dir2 = "dataset/spectrograms/sawing/"

   in_file = str(input('enter file name: '))

   start = time.time()
   res = predict_spectrogram(in_dir1 + in_file, model)
   print(res)
   print('prediction time: ', time.time() - start)
