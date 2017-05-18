import logging
import numpy as np
from collections import Counter
import itertools
import torch.nn as nn
import torch
import code
import os
from att_models import *
from six.moves import cPickle

def load_data(in_file):
    sentences = []
    num_examples = 0
    f = open(in_file, 'r')
    line = f.readline()
    while line != "": 
        line = line.strip().split("\t")
        sentences.append([line[0].split(), line[1].split()])
        line = f.readline()
    f.close()
    return sentences

def flatten(l):
    res = []
    for e in l:
        if str(type(e)) == "<type 'list'>":
            res += flatten(e)
        else:
            res.append(e)
    return res

def build_dict(sentences, max_words=50000):
    word_count = Counter()
    sentences = flatten(sentences)
    # code.interact(local=locals())
    for w in sentences:
        word_count[w] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 1
    return {w[0]: index+1 for (index, w) in enumerate(ls)}, total_words

def load_dict(file):
    word_dict = {}
    i = 0
    with open(file, "r") as f:
        for line in f:
            word_dict[line.strip()] = i
            i += 1
    return word_dict, i

def encode(sentences, word_dict, sort_by_len=True):
    '''
        Encode the sequences. 
    '''
    out_sentences = []

    for idx, sentence in enumerate(sentences):
        seq0 = [word_dict[w] if w in word_dict else 0 for w in sentence[0]]
        seq1 = [word_dict[w] if w in word_dict else 0 for w in sentence[1]]
        if len(seq0) > 0:
            out_sentences.append([seq0, seq1])

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x][0]))
       
    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(out_sentences)
        out_sentences = [out_sentences[i] for i in sorted_index]
    return out_sentences

def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def prepare_data(seqs):
    lengths0 = [len(seq[0]) for seq in seqs]
    lengths1 = [len(seq[1]) for seq in seqs]
    n_samples = len(seqs)
    max_len0 = np.max(lengths0)
    max_len1 = np.max(lengths1)
    x = np.zeros((n_samples, max_len0)).astype('int32')
    x_mask = np.zeros((n_samples, max_len0)).astype('float32')
    y = np.zeros((n_samples, max_len1)).astype('int32')
    y_mask = np.zeros((n_samples, max_len1)).astype('float32')
    for idx, seq in enumerate(seqs):
        x[idx, :lengths0[idx]] = seq[0]
        x_mask[idx, :lengths0[idx]] = 1.0
        y[idx, :lengths1[idx]] = seq[1]
        y_mask[idx, :lengths1[idx]] = 1.0
    return x, x_mask, y, y_mask
  


def gen_examples(sentences, batch_size):
    minibatches = get_minibatches(len(sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_sentences = [sentences[t] for t in minibatch]
        mb_x, mb_x_mask, mb_y, mb_y_mask = prepare_data(mb_sentences)
        all_ex.append((mb_x, mb_x_mask, mb_y, mb_y_mask))
    return all_ex


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class HingeModelCriterion(nn.Module):
    def __init__(self):
        super(HingeModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = input.view(input.size(0)*input.size(1), input.size(2))
        target = target.view(target.size(0)*target.size(1), 1)
        correct = input.gather(1, target).expand_as(input)
        loss = torch.sum(torch.sum(torch.max(input + 1  - correct, 0)[0], 1) - 1) / torch.sum(mask)

        return loss
def model_setup(args):
    if args.model == "AttentionEncoderDecoderModel":
        model = AttentionEncoderDecoderModel(args)  
    else:
        raise Exception("Model not supported: {}".format(args.model))

    if os.path.isdir(args.checkpoint_path):
        if args.load_best:
            if os.path.isfile(os.path.join(args.checkpoint_path,"infos.pkl")):
                print('load model from', os.path.join(args.checkpoint_path, 'model_best.pth'))
                model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'model_best.pth')))
        else:
            if os.path.isfile(os.path.join(args.checkpoint_path,"infos_best.pkl")):
                print('load model from', os.path.join(args.checkpoint_path, 'model.pth'))
                model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'model.pth')))

    return model
        
