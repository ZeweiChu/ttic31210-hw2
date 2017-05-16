import sys, os
import time
import seq2seq_utils as utils
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


def eval(model, data, args, crit, err=None):
	total_dev_batches = len(data)
	correct_count = 0.
	# bar = progressbar.ProgressBar(max_value=total_dev_batches).start()
	loss = 0.
	total_num_words = 0.

	print("total %d" % total_dev_batches)
	total_num_words = 0.
	for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(data):

		batch_size = mb_x.shape[0]
		mb_x = Variable(torch.from_numpy(mb_x)).long()
		mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
		hidden = model.init_hidden(batch_size)
		mb_input = Variable(torch.from_numpy(mb_y[:,:-1])).long()
		mb_out = Variable(torch.from_numpy(mb_y[:, 1:])).long()
		mb_out_mask = Variable(torch.from_numpy(mb_y_mask[:, 1:]))
		
		mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)
		num_words = torch.sum(mb_out_mask).data[0]
		loss += crit(mb_pred, mb_out, mb_out_mask).data[0] * num_words

		total_num_words += num_words

		mb_pred = torch.max(mb_pred.view(mb_pred.size(0) * mb_pred.size(1), mb_pred.size(2)), 1)[1]
		correct = (mb_pred == mb_out).float()
		correct = correct * mb_out_mask.view(mb_out_mask.size(0) * mb_out_mask.size(1), 1)
		if err != None:
			corr = correct.data.numpy().reshape(-1)
			mb_pred = mb_pred.data.numpy().reshape(-1)
			mb_out = mb_out.data.numpy().reshape(-1)
			for i in range(corr.shape[0]):
				if corr[i] == 0:
					# if "<" + str(mb_out[i]) + "," + str(mb_pred[i]) + ">" in err:
					err[str(mb_out[i]) + "," + str(mb_pred[i])] += 1
		
		# code.interact(local=locals())
		correct_count += torch.sum(correct * mb_out_mask.contiguous().view(mb_out_mask.size(0) * mb_out_mask.size(1), 1)).data[0]
		# bar.update(idx+1)

	# bar.finish()
	return correct_count, loss, total_num_words

def train(sentences):
	pass


def main(args):

	train_sentences = utils.load_data(args.train_file)
	dev_sentences = utils.load_data(args.dev_file)

	args.num_train = len(train_sentences)
	args.num_dev = len(dev_sentences)

	word_dict, args.vocab_size = utils.load_dict(args.vocab_file)
	# word_dict, args.vocab_size = utils.build_dict(train_sentences, max_words=args.vocab_size)
	# word_dict["UNK"] = 0
	


	train_sentences = utils.encode(train_sentences, word_dict)

	train_sentences = utils.gen_examples(train_sentences, args.batch_size)

	
	dev_sentences = utils.encode(dev_sentences, word_dict)
	dev_sentences = utils.gen_examples(dev_sentences, args.batch_size)

	# code.interact(local=locals())


	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)
	elif args.model == "EncoderDecoderModel":
		model = EncoderDecoderModel(args)
	elif args.model == "BiEncoderDecoderModel":
		model = BiEncoderDecoderModel(args)
	elif args.model == "BOWEncoderDecoderModel":
		model = BOWEncoderDecoderModel(args)



	if args.criterion == "HingeModelCriterion":
		crit = utils.HingeModelCriterion()
	elif args.criterion == "LanguageModelCriterion":
		crit = utils.LanguageModelCriterion()

	print("start evaluating on dev...")
	
	err = Counter()
	correct_count, loss, num_words = eval(model, dev_sentences, args, crit, err=err)
	if err != None:
		err = err.most_common()[:20]
		word_dict_rev = {v: k for k, v in word_dict.iteritems()}
		for pair in err:
			p = pair[0].split(",")
			pg = word_dict_rev[int(p[0])]
			pp = word_dict_rev[int(p[1])]
			print("ground truth: " + pg + ", predicted: " + pp + ", number: " + str(pair[1]) + "\\\\")
		# code.interact(local=locals())

	loss = loss / num_words
	acc = correct_count / num_words
	print("dev loss %s" % (loss) )
	print("dev accuracy %f" % (acc))
	print("dev total number of words %f" % (num_words))
	best_acc = acc
	prev_acc = acc

	learning_rate = args.learning_rate
	if args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	elif args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	# best_loss = loss

	# F.linear(embedded.view(embedded.size(0)*embedded.size(1), embedded.size(2)), model.embed.weight)

	total_num_sentences = 0.
	total_time = 0.
	for epoch in range(args.num_epoches):

		np.random.shuffle(train_sentences)
		num_batches = len(train_sentences)
		# bar = progressbar.ProgressBar(max_value= num_batches * args.eval_epoch, redirect_stdout=True)
		total_train_loss = 0.
		total_num_words = 0.
		
		start = time.time()
		for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in tqdm(enumerate(train_sentences)):

			batch_size = mb_x.shape[0]
			total_num_sentences += batch_size
			mb_x = Variable(torch.from_numpy(mb_x)).long()
			mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
			hidden = model.init_hidden(batch_size)
			mb_input = Variable(torch.from_numpy(mb_y[:,:-1])).long()
			mb_out = Variable(torch.from_numpy(mb_y[:, 1:])).long()
			mb_out_mask = Variable(torch.from_numpy(mb_y_mask[:, 1:]))
			
			mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)

			loss = crit(mb_pred, mb_out, mb_out_mask)
			num_words = torch.sum(mb_out_mask).data[0]
			total_train_loss += loss.data[0] * num_words
			total_num_words += num_words
	
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		end = time.time()
		total_time += (end - start)
		print("training loss: %f" % (total_train_loss / total_num_words))
		

		

		# code.interact(local=locals())
		if (epoch+1) % args.eval_epoch == 0:
			

			print("start evaluating on dev...")
	
			correct_count, loss, num_words = eval(model, dev_sentences, args, crit)

			loss = loss / num_words
			acc = correct_count / num_words
			print("dev loss %s" % (loss) )
			print("dev accuracy %f" % (acc))
			print("dev total number of words %f" % (num_words))

			if acc > best_acc:
				torch.save(model, args.model_file)
				best_acc = acc
				# infos['epoch'] = epoch
				# infos['best_acc'] = best_acc
				# infos['vocab']

				print("model saved...")
			elif acc < prev_acc:
				learning_rate *= 0.5
				if args.optimizer == "SGD":
					optimizer = optim.SGD(model.parameters(), lr=learning_rate)
				elif args.optimizer == "Adam":
					optimizer = optim.Adam(model.parameters(), lr=learning_rate)
			prev_acc = acc

			print("best dev accuracy: %f" % best_acc)
			print("#" * 60)


	test_sentences = utils.load_data(args.test_file)
	args.num_test = len(test_sentences)
	test_sentences = utils.encode(test_sentences, word_dict)
	test_sentences = utils.gen_examples(test_sentences, args.batch_size)
	correct_count, loss, num_words = eval(model, test_sentences, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("test loss %s" % (loss) )
	print("test accuracy %f" % (acc))
	print("test total number of words %f" % (num_words))

	print("#sents/sec: %f" % (total_num_sentences/total_time) )

	correct_count, loss, num_words = eval(model, train_sentences, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("train loss %s" % (loss) )
	print("train accuracy %f" % (acc))


if __name__ == "__main__":
	args = config.get_args()
	main(args)
