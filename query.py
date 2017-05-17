import sys, os
import time
import seq2seq_utils
import utils
import config
import logging
import code
import numpy as np
from models import *
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import MSELoss
import progressbar
from tqdm import tqdm
import pickle
import math
from collections import Counter
import time


def main(args):

	train_sentences_raw = seq2seq_utils.load_data(args.train_file)
	dev_sentences_raw = utils.load_data(args.dev_file)
	args.num_train = len(train_sentences_raw)
	args.num_dev = len(dev_sentences_raw)
	word_dict, args.vocab_size = utils.load_dict(args.vocab_file)
	train_sentences = seq2seq_utils.encode(train_sentences_raw, word_dict)
	train_sentences = seq2seq_utils.gen_examples(train_sentences, args.batch_size)
	dev_sentences = utils.encode(dev_sentences_raw, word_dict)
	dev_sentences = utils.gen_examples(dev_sentences, args.batch_size)


	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)

	train_encodings = []

	for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(train_sentences):
		batch_size = mb_x.shape[0]
		mb_x = Variable(torch.from_numpy(mb_x)).long()
		mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
		hidden = model.init_hidden(batch_size)
		hiddens, _ = model.encode(mb_x, mb_x_mask, hidden)
		train_encodings.append(hiddens)

	train_encodings = torch.cat(train_encodings, 0)

	# code.interact(local=locals())

	idx = 0
	mb_x, mb_x_mask = dev_sentences[0]
	batch_size = mb_x.shape[0]
	mb_x = Variable(torch.from_numpy(mb_x)).long()
	mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
	hidden = model.init_hidden(batch_size)
	hiddens, _ = model.encode(mb_x, mb_x_mask, hidden)
	similarity = F.linear(hiddens, train_encodings) 
	# code.interact(local=locals())
	similarity = similarity / torch.norm(hiddens, 2, 1).expand_as(similarity)
	similarity = similarity / torch.norm(train_encodings, 2, 1).transpose(1,0).expand_as(similarity)
	similarity = similarity.data.numpy()
	nearest = similarity.argpartition(10)[:,-10:]
	for i in range(10):
		# code.interact(local=locals())
		print("dev: " + " ".join(dev_sentences_raw[idx*args.batch_size + i]))
		for k in range(10):
			print("nearest neighbor in training: " + " ".join(train_sentences_raw[nearest[i][k]][0]) + "\\\\")
		
			

		
		# code.interact(local=locals())

	


if __name__ == "__main__":
	args = config.get_args()
	main(args)
