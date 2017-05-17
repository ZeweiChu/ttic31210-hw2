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

	dev_sentences_raw = utils.load_data(args.dev_file)
	args.num_dev = len(dev_sentences_raw)
	word_dict, args.vocab_size = utils.load_dict(args.vocab_file)
	dev_sentences = utils.encode(dev_sentences_raw, word_dict)
	dev_sentences = utils.gen_examples(dev_sentences, args.batch_size)


	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)

	# code.interact(local=locals())

	idx = 0
	mb_x, mb_x_mask = dev_sentences[0]
	batch_size = mb_x.shape[0]
	mb_x = Variable(torch.from_numpy(mb_x)).long()
	mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
	hidden = model.init_hidden(batch_size)
	hiddens, cellgates = model.encode(mb_x, mb_x_mask, hidden)

	mb_y = Variable(torch.zeros(batch_size, 1)).long()
	generated = model.decode(mb_y, hiddens, cellgates).data.numpy()
	B, T = generated.shape
	
	word_dict_rev = {v: k for k, v in word_dict.iteritems()}
	for i in range(B):
		print("dev: " + " ".join(dev_sentences_raw[i]).replace("<", "$<$").replace(">", "$>$") + "\\\\")
		res = []
		for j in range(T):

			res.append(word_dict_rev[generated[i][j]])
			if word_dict_rev[generated[i][j]] == "<\s>":
				break
		print("generated: " + " ".join(res).replace("<", "$<$").replace(">", "$>$") + "\\\\")
		

	# code.interact(local=locals())
		# code.interact(local=locals())

	


if __name__ == "__main__":
	args = config.get_args()
	main(args)
