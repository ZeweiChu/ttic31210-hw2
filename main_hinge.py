import sys, os
import time
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
from tqdm import tqdm
import progressbar
import pickle
import math
import time

def eval(model, data, args, crit):
	total_dev_batches = len(data)
	correct_count = 0.
	# bar = progressbar.ProgressBar(max_value=total_dev_batches).start()
	loss = 0.
	total_num_words = 0.

	print("total %d" % total_dev_batches)
	for idx, (mb_s, mb_mask) in enumerate(data):
		batch_size = mb_s.shape[0]
		mb_input = Variable(torch.from_numpy(mb_s[:,:-1]), volatile=True).long()
		mb_out = Variable(torch.from_numpy(mb_s[:, 1:]), volatile=True).long()
		mb_out_mask = Variable(torch.from_numpy(mb_mask[:, 1:]), volatile=True)
		hidden = model.init_hidden(batch_size)

		if args.model == "LSTMHingeOutEmbNegModel":
			mb_pred, hidden = model.predict(mb_input, hidden)
			num_words = torch.sum(mb_out_mask).data[0]
			loss += crit(mb_pred, mb_out, mb_out_mask).data[0] * num_words
		else:
			mb_pred, hidden = model(mb_input, hidden)
			num_words = torch.sum(mb_out_mask).data[0]
			loss += crit(mb_pred, mb_out, mb_out_mask).data[0] * num_words
		total_num_words += num_words

		mb_pred = torch.max(mb_pred.view(mb_pred.size(0) * mb_pred.size(1), mb_pred.size(2)), 1)[1]
		correct = (mb_pred == mb_out).float()
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
	

	# pickle.dump(word_dict, open(args.dict_file, "wb"))

	train_sentences = utils.encode(train_sentences, word_dict)
	train_sentences = utils.gen_examples(train_sentences, args.batch_size)

	
	dev_sentences = utils.encode(dev_sentences, word_dict)
	dev_sentences = utils.gen_examples(dev_sentences, args.batch_size)

	# code.interact(local=locals())
	
	att_dict = {}

	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)
	elif args.model == "LSTMHingeModel":
		model = LSTMHingeModel(args)
	elif args.model == "LSTMHingeOutEmbModel":
		model = LSTMHingeOutEmbModel(args)
	elif args.model == "LSTMHingeOutEmbNegModel":
		model = LSTMHingeOutEmbNegModel(args)
	elif args.model == "LSTMModel":
		model = LSTMModel(args)


	if args.criterion == "HingeModelCriterion":
		crit = utils.HingeModelCriterion()
	elif args.criterion == "LanguageModelCriterion":
		crit = utils.LanguageModelCriterion()

	print("start evaluating on dev...")
	
	correct_count, loss, num_words = eval(model, dev_sentences, args, crit)

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
	flog = open(args.log_file, "w")

	total_num_sentences = 0.
	total_time = 0.
	for epoch in range(args.num_epoches):

		np.random.shuffle(train_sentences)
		num_batches = len(train_sentences)
		total_train_loss = 0.
		total_num_words = 0.
		start = time.time()
		for idx, (mb_s, mb_mask) in tqdm(enumerate(train_sentences)):

			batch_size = mb_s.shape[0]
			total_num_sentences += batch_size
			mb_input = Variable(torch.from_numpy(mb_s[:,:-1])).long()
			mb_out = Variable(torch.from_numpy(mb_s[:, 1:])).long()
			mb_out_mask = Variable(torch.from_numpy(mb_mask[:, 1:]))
			hidden = model.init_hidden(batch_size)
			if args.model == "LSTMHingeOutEmbNegModel":
				
				mb_pred, hidden = model(mb_input, hidden, mb_out)
				mb_out = Variable(mb_pred.data.new(mb_pred.size(0), mb_pred.size(1)).zero_()).long()
				loss = crit(mb_pred, mb_out, mb_out_mask)
			else:
				mb_pred, hidden = model(mb_input, hidden)
				loss = crit(mb_pred, mb_out, mb_out_mask)
			num_words = torch.sum(mb_out_mask).data[0]
			total_train_loss += loss.data[0] * num_words
			# code.interact(local=locals())
			total_num_words += num_words
	
			optimizer.zero_grad()
			loss.backward()

			nn.utils.clip_grad_norm(model.parameters(), args.grad_clipping)
			optimizer.step()
		
		end = time.time()
		total_time += (end - start)
		
		print("training loss: %f" % (total_train_loss / total_num_words))

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

			flog.write("%f\t%f\t%f\t%f\t%f\n"%(total_time, total_num_sentences, best_acc, acc, loss))



	correct_count, loss, num_words = eval(model, train_sentences, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("train loss %s" % (loss) )
	print("train accuracy %f" % (acc))
	print("#sents/sec: %f" % (total_num_sentences/total_time) )

	model = torch.load(args.model_file)
	test_sentences = utils.load_data(args.test_file)
	args.num_test = len(test_sentences)
	test_sentences = utils.encode(test_sentences, word_dict)
	test_sentences = utils.gen_examples(test_sentences, args.batch_size)
	correct_count, loss, num_words = eval(model, test_sentences, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("test loss %s" % (loss) )
	print("test accuracy %f" % (acc))


	flog.close()

if __name__ == "__main__":
	args = config.get_args()
	main(args)
