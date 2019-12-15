import sys,os
from collections import Counter


def get_all_token(train_dict, val_dict):
	descriptions = list(train_dict.values()) + list(val_dict.values())
	token = list()
	for descs in descriptions:
		for desc in descs:
			token += desc.split()
	return token

def get_vocabulary(train_dict, val_dict):
	token = get_all_token(train_dict, val_dict)
	types = list(set(token))
	return types

def index_vocab(vocab):
  """
  Assign idx to word and vice versa
  """
  idx2word, word2idx = dict(), dict()
  for idx, word in enumerate(vocab):
    idx2word[idx] = word
    word2idx[word] = idx
  return idx2word, word2idx


def filter_freq(token_list, min_freq=10):
  """
  #Consider only words which occur at least min_freq x
  """
  token2count = Counter(token_list)
  vocab = [w for w,count in token2count.items() if count >= min_freq]
  return vocab


def get_max_length(train_dict, val_dict):
	"""
	determine the maximum sequence length
	"""
	descs = list()
	for val in list(train_dict.values()) + list(val_dict.values()):
		descs += val
	return max(len(desc.split()) for desc in descs)