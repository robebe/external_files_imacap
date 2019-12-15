import sys,os
import pickle
import numpy as np


def _word_to_embed(vec_file, outf):
  """
  map pretrained embedding to word and save to outf
  """
  word2embed = dict()
  with open(vec_file) as inf:
    for line in inf.readlines()[1:]:#skip first line as it contains no embedding
      tmp_vals = line.rstrip().rsplit(" ")
      word = tmp_vals[0]
      coefs = np.asarray(tmp_vals[1:], dtype='float32')
      word2embed[word] = coefs
  with open(outf, "wb") as out:
   pickle.dump(word2embed, out)
  return word2embed

def get_word_to_embed(vec_file, dict_file):
  if os.path.isfile(dict_file):
    with open(dict_file, "rb") as inf:
      return pickle.load(inf)
  else:
    return _word_to_embed(vec_file=vec_file, outf=dict_file)

def get_embed_mtx(vocab_size, embed_dim, word2idx, vec_dict):
  """
  Get embed_dim dense vector for each of the vocabulary
  """
  not_found = 0
  embed_mtx = np.zeros((vocab_size, embed_dim))
  for word, idx in word2idx.items():
    try:
      embed_mtx[idx] = vec_dict[word]
    except KeyError:
      not_found +=1
  print("{} out of {} not found in pretrained word vectors".format(not_found, len(word2idx)))
  return embed_mtx