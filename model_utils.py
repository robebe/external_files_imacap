import sys,os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(idx2descs, idx2image_feature, word2idx, max_length, num_photos_per_batch, vocab_size):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in idx2descs.items():
            n+=1
            # retrieve the photo feature
            image_feature = idx2image_feature[key]
            for desc in desc_list:
                # encode the sequence
                seq = [word2idx[word] for word in desc.split(' ') if word in word2idx.keys()]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    former_seq, pred_seq = seq[:i], seq[i]
                    # pad input sequence
                    former_seq = pad_sequences([former_seq], maxlen=max_length)[0]
                    # encode output sequence
                    pred_seq = to_categorical([pred_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(image_feature)
                    X2.append(former_seq)
                    y.append(pred_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

"""
  ret = [(token, last_encoded_layer[i]) for i, token in enumerate(tokenized_text)]
  return ret#return list of tuples with token and embedding

#example usage
#embedding_tuples = get_bert_sentence_encoding("Das ist ein Beispielsatz.")
#print(embedding_tuples[0])
#context_de_model.fit_generator(context_de_generator, epochs=epochs, steps_per_epoch=de_steps, verbose=1)
"""
"""
def contextualised_data_generator(idx2descs, idx2image_feature, word2idx, max_length, num_photos_per_batch, vocab_size):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in idx2descs.items():
            n+=1
            # retrieve the photo feature
            image_feature = idx2image_feature[key]
            for desc in desc_list:
                # encode the sequence
                #seq = [word2idx[word] for word in desc.split(' ') if word in word2idx.keys()]
                seq = get_bert_sentence_encoding(desc)
                # split one sequence into multiple X, y pairs
                #for i in range(1, len(seq)):
                for i in range(1, len(seq)):
                    # split into input and output pair
                    former_seq, pred_seq = seq[:i], seq[i]
                    # pad input sequence
                    former_seq = pad_sequences([former_seq], maxlen=max_length)[0]
                    # encode output sequence
                    pred_seq = to_categorical([pred_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(image_feature)
                    X2.append(former_seq)
                    y.append(pred_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0
"""

def greedySearch(model, encoded_image, max_length, word2idx, idx2word, start_seq_token, end_seq_token):
  produced_seq = [start_seq_token]
  for i in range(max_length):
      sequence = [word2idx[w] for w in produced_seq if w in word2idx]
      encoded_sequence = pad_sequences([sequence], maxlen=max_length)
      prediction = np.argmax(model.predict([encoded_image, encoded_sequence], verbose=0))
      produced_seq.append(idx2word[prediction])
      if idx2word[prediction] == end_seq_token:
          break
  #return ' '.join(produced_seq[1:-1])
  return ' '.join(produced_seq)

def _filter_finished_hypos(hypothesis, word2idx, end_seq_token):
  finished_hypos = list()
  for score, hypo in hypothesis:
    if hypo[-1] == word2idx[end_seq_token]:
      finished_hypos.append((score, hypo))
      hypothesis.remove((score, hypo))
  return finished_hypos, hypothesis

import math

def beamSearch(model, encoded_image, max_length, word2idx, idx2word, start_seq_token, end_seq_token, beam=5):
  #https://stackoverflow.com/questions/50826625/python-beam-search-for-keras-lstm-model-generating-the-same-sequence
  hypothesis = [(1.0, [word2idx[start_seq_token]])]
  finished_hypos = list()
  for cur_position in range(max_length):
    if len(finished_hypos) < beam:
      tmp_beams = list()
      for cur_score, cur_hypo in hypothesis:
        encoded_cur_ids = pad_sequences([cur_hypo], maxlen=max_length)
        predicted = model.predict([encoded_image, encoded_cur_ids], verbose=0)[0]
        top_k_indices = predicted.argsort()[-beam:][::-1]
        for tmp_top_idx in top_k_indices:
          tmp_score = cur_score * (- math.log(predicted[tmp_top_idx]))
          tmp_hypo = cur_hypo + [tmp_top_idx]
          tmp_beams.append((tmp_score, tmp_hypo))
      hypothesis = sorted(tmp_beams, key=lambda x: x[0])[-beam:]
      tmp_finished_hypos, hypothesis = _filter_finished_hypos(hypothesis, word2idx, end_seq_token)
      finished_hypos += tmp_finished_hypos
  final_seq_indices = sorted(hypothesis+finished_hypos, key=lambda x: x[0])[0][1]
  print("this is possibly not possible")
  return " ".join([idx2word[idx] for idx in final_seq_indices])
