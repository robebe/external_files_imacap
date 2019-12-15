import sys,os
import string

start_seq_token = "<s>"
end_seq_token = "</s>"

def _clear_string(s):
	tokens = [t.lower() for t in s.strip().split()]
	#remove punctuation
	table = str.maketrans("", "", string.punctuation)
	no_puncts = [tok.translate(table) for tok in tokens]
	#remove single letters (assuming no pronouns like 'I')
	only_words = [tok for tok in no_puncts if len(tok) > 1]
	#remove numbers
	return start_seq_token + " " + " ".join([tok for tok in only_words if tok.isalpha()]) + " " + end_seq_token

def load_file(token_fname, image_fname):
	id2descs = dict()
	with open(token_fname, "r") as tfile,\
				open(image_fname, "r") as ifile:
		descriptions = [_clear_string(desc) for desc in tfile.readlines()]
		image_ids = [i.rstrip(".jpg\n") for i in ifile.readlines()]
	assert(len(descriptions) == len(image_ids))
	for i, desc in zip(image_ids, descriptions):
		try:#append multiple descriptions to image id
			id2descs[i].append(desc)
		except KeyError:#image id not yet in dictionary
			id2descs[i] = [desc]
	return id2descs