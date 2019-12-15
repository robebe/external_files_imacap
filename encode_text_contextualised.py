import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
bert_model.eval()
bert_model.to("cuda")

bert_start = "[CLS]"
bert_end = "[SEP]"
bert_vocab_size = 119547
bert_embedding_size = 768

def get_bert_sentence_encoding(sentence_string):
  """
  Input: sentence to be encoded. return last layer encoding for each token in the sentence
  """
  sentence = bert_start + " " + sentence_string + " " + bert_end

  tokenized_text = bert_tokenizer.tokenize(sentence)
  indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
  segment_ids = [0]*len(tokenized_text)

  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segment_ids])

  tokens_tensor = tokens_tensor.to('cuda')
  segments_tensors = segments_tensors.to('cuda')

  with torch.no_grad():
      encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
  
  last_encoded_layer = encoded_layers[0][0]

  ret = list()
  ret = [(token, last_encoded_layer[i]) for i, token in enumerate(tokenized_text)]
  return ret#return list of tuples with token and embedding

#example usage
#embedding_tuples = get_bert_sentence_encoding("Das ist ein Beispielsatz.")
#print(embedding_tuples[0])