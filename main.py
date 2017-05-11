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
import progressbar
import pickle
import math

def eval(model, data, args, att_dict={}):
	total_dev_batches = len(data)
	correct_count = 0.
	bar = progressbar.ProgressBar(max_value=total_dev_batches).start()
	loss = 0.

	print("total dev %d" % total_dev_batches)

	# code.interact(local=locals())
	for idx, (mb_d, mb_mask_d, mb_l) in enumerate(data):
		# print(mb_d)
		mb_d = Variable(torch.from_numpy(mb_d)).long()
		# print("after")
		# print(mb_d)
		mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
		if args.check_att:
			mb_out, att_dict = model(mb_d, mb_mask_d, att_dict, check_att=args.check_att)
		else:
			mb_out = model(mb_d, mb_mask_d, att_dict)
		# print(mb_out)

		batch_size = mb_d.size(0)
		mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
		loss += (- mb_a * torch.log(mb_out + 1e-9) - (1. - mb_a) * torch.log(1. - mb_out + 1e-9)).sum().data[0]
		# (torch.abs(mb_a - mb_out) * torch.log(torch.abs(mb_a - mb_out) + 1e-9)).sum().data[0]
		
		res = torch.abs(mb_a - mb_out) < 0.5
		# code.interact(local=locals())
		correct_count += res.sum().data[0]
		# print(correct_count)

		bar.update(idx+1)

	bar.finish()
	
	if args.check_att:
		return correct_count, loss, att_dict
	else:
		return correct_count, loss

def train(sentences):
	pass


def main(args):

	train_sentences = utils.load_data(args.train_file)
	dev_sentences = utils.load_data(args.dev_file)

	args.num_train = len(train_sentences)
	args.num_dev = len(dev_sentences)

	word_dict, args.vocab_size = utils.build_dict(train_sentences, max_words=args.vocab_size)
	word_dict["UNK"] = 0
	

	# pickle.dump(word_dict, open(args.dict_file, "wb"))

	train_sentences = utils.encode(train_sentences, word_dict)
	train_sentences = utils.gen_examples(train_sentences, args.batch_size)

	
	dev_sentences = utils.encode(dev_sentences, word_dict)
	dev_sentences = utils.gen_examples(dev_sentences, args.batch_size)

	# code.interact(local=locals())
	
	att_dict = {}

	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)
	else:
		model = LSTMModel(args)

	# if args.test_only:
	# 	print("start evaluating on test")
	# 	correct_count, loss = eval(model, all_test, args)
	# 	print("test accuracy %f" % (float(correct_count) / float(args.num_test)))
	# 	loss = loss / args.num_test
	# 	print("test loss %f" % loss)

	# 	correct_count, loss = eval(model, all_dev, args)
	# 	print("dev accuracy %f" % (float(correct_count) / float(args.num_dev)))
	# 	loss = loss / args.num_dev
	# 	print("dev loss %f" % loss)
	# 	return 0

	# correct_count, loss = eval(model, all_dev, args)
	# acc = float(correct_count) / float(args.num_dev)
	# best_acc = acc
	# print("dev accuracy %f" % acc)
	# loss = loss / args.num_dev
	# print("dev loss %f" % loss)
	# code.interact(local=locals())

	learning_rate = args.learning_rate
	if args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	elif args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	# best_loss = loss


	crit = utils.LanguageModelCriterion()
	for epoch in range(args.num_epoches):

		np.random.shuffle(train_sentences)
		num_batches = len(train_sentences)
		# bar = progressbar.ProgressBar(max_value= num_batches * args.eval_epoch, redirect_stdout=True)
		total_train_loss = 0.
		total_num_words = 0.
		for idx, (mb_s, mb_mask) in enumerate(train_sentences):

			batch_size = mb_s.shape[0]
			mb_input = Variable(torch.from_numpy(mb_s[:,:-1])).long()
			mb_out = Variable(torch.from_numpy(mb_s[:, 1:])).long()
			mb_out_mask = Variable(torch.from_numpy(mb_mask[:, 1:]))
			hidden = model.init_hidden(batch_size)
			mb_pred, hidden = model(mb_input, hidden)

			loss = crit(mb_pred, mb_out, mb_out_mask)
			num_words = torch.sum(mb_out_mask).data[0]
			total_train_loss += loss.data[0] * num_words
			total_num_words += num_words
			# loss = (torch.abs(mb_a - mb_out) * torch.log(torch.abs(mb_a - mb_out) + 1e-9)).sum() / batch_size
		
			optimizer.zero_grad()
			loss.backward()
			# for p in model.parameters():
			# 	grad = p.grad.data.numpy()
			# 	grad[np.isnan(grad)] = 0
			# 	p.grad.data = torch.Tensor(grad)
			optimizer.step()
			print(loss.data[0])
			# bar.update(num_batches * (epoch % args.eval_epoch) + idx +1)
		
		# bar.finish()
		print("training loss: %f" % (total_train_loss / total_num_words))

		if (epoch+1) % args.eval_epoch == 0:
			

			print("start evaluating on dev...")
			# print(all_dev)
			correct_count, loss = eval(model, dev_sentences, args)
			# print("correct count %f" % correct_count)
			# print("total count %d" % args.num_dev)
			acc = float(correct_count) / float(args.num_dev)
			print("dev accuracy %f" % acc)
			loss = loss / args.num_dev
			print("dev loss %f" % loss)
			

			if acc > best_acc:
				torch.save(model, args.model_file)
				best_acc = acc
				print("model saved...")
			else:
				learning_rate *= 0.5
				if args.optimizer == "SGD":
					optimizer = optim.SGD(model.parameters(), lr=learning_rate)
				elif args.optimizer == "Adam":
					optimizer = optim.Adam(model.parameters(), lr=learning_rate)

			print("best dev accuracy: %f" % best_acc)
			print("#" * 60)


if __name__ == "__main__":
	args = config.get_args()
	main(args)
