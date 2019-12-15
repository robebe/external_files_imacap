import sys, os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import pickle

def _convert_image(image_path):
  """
  Convert all the images to size 299x299 as expected by the inception v3 model
  Convert PIL image to numpy array of 3-dimensions
  Add one more dimension
  preprocess the images using preprocess_input() from inception module
  """
  img = image.load_img(image_path, target_size=(299, 299))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return x

def _encode(image, model):
  """
  encode a given image into a vector of size (2048, )
  """
  image = _convert_image(image) # preprocess the image
  fea_vec = model.predict(image) # Get the encoding vector for the image
  fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
  return fea_vec

def encode_and_store_images(model, image_path, base_path, to_encode, outf):
  #image_files = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
  id2encoding = dict()
  
  for fname in to_encode:
    id2encoding[fname.rstrip(".jpg\n")] = _encode(os.path.join(image_path, fname), model)
  with open(os.path.join(base_path, outf), "wb") as out:
    pickle.dump(id2encoding, out)
  return id2encoding