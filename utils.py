import logging
import numpy as np
from collections import Counter
import itertools
import torch.nn as nn
import torch


def load_data(in_file):
    sentences = []
    num_examples = 0
    f = open(in_file, 'r')
    line = f.readline()
    while line != "": 
        sentences.append(line.strip().split() )
        line = f.readline()
    f.close()
    return sentences

def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            word_count[w] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 1
    return {w[0]: index+1 for (index, w) in enumerate(ls)}, total_words

def encode(sentences, word_dict, sort_by_len=True):
    '''
        Encode the sequences. 
    '''
    out_sentences = []

    for idx, sentence in enumerate(sentences):
        seq = [word_dict[w] if w in word_dict else 0 for w in sentence]
        if len(seq) > 0:
            out_sentences.append(seq)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

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
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype('float32')
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask

def gen_examples(sentences, batch_size):
    minibatches = get_minibatches(len(sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_sentences = [sentences[t] for t in minibatch]
        mb_s, mb_mask = prepare_data(mb_sentences)
        all_ex.append((mb_s, mb_mask))
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
