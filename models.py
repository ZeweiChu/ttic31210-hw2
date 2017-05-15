import torch
import torch.nn as nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        num_directions = 2 if bidirectional else 1

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                gate_size = 4 * hidden_size

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                setattr(self, weights[0], w_ih)
                setattr(self, weights[1], w_hh)
                if bias:
                    setattr(self, weights[2], b_ih)
                    setattr(self, weights[3], b_hh)
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        B, T, input_encoding_size = input.size()

        forgetgates = []
        hiddens = []
        cellgates = []
        outgates = []

        hx, cx = hx
        for i in range(T):
            gates = F.linear(input[:, i, :], self.weight_ih_l0, self.bias_ih_l0) \
                        + F.linear(hx, self.weight_hh_l0, self.bias_hh_l0)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            forgetgates.append(forgetgate)
            cellgate = F.tanh(cellgate)
            cellgates.append(cellgate.unsqueeze(1))
            outgate = F.sigmoid(outgate)
            outgates.append(outgate)

            cx = (forgetgate * cx) + (ingate * cellgate)
            hx = outgate * F.tanh(cx)
            hiddens.append(hx.unsqueeze(1))

        hiddens = torch.cat(hiddens, 1)
        cellgates = torch.cat(cellgates, 1)
        return forgetgates, hiddens, cellgates, outgate

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()

        self.nhid = args.hidden_size
        self.nlayers = args.num_layers

        self.decoder = nn.Linear(self.nhid, args.vocab_size)
        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.lstm = LSTM(args.embedding_size, args.hidden_size)

        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
        		Variable(weight.new(bsz, self.nhid).zero_()))


    def forward(self, d, hidden):
        d_embedded = self.embed(d)
        forgetgates, hiddens, cellgates, output = self.lstm(d_embedded, hidden)
        decoded = self.decoder(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens

class LSTMHingeModel(nn.Module):
    def __init__(self, args):
        super(LSTMHingeModel, self).__init__()

        self.nhid = args.hidden_size
        self.nlayers = args.num_layers

        # self.decoder = nn.Linear(self.nhid, args.vocab_size)
        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.lstm = LSTM(args.embedding_size, args.hidden_size)

        self.embed.weight.data.uniform_(-0.1, 0.1)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))


    def forward(self, d, hidden):
        d_embedded = self.embed(d)
        forgetgates, hiddens, cellgates, output = self.lstm(d_embedded, hidden) # hiddens: B * T * hidden_size
        decoded = F.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)), self.embed.weight) 
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens # decoded: B * T * vocab_size


class LSTMHingeOutEmbModel(nn.Module):
    def __init__(self, args):
        super(LSTMHingeOutEmbModel, self).__init__()
        self.nhid = args.hidden_size
        self.nlayers = args.num_layers
        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.lstm = LSTM(args.embedding_size, args.hidden_size)
        self.out_embed = Parameter(torch.Tensor(args.vocab_size, args.hidden_size))
        
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.out_embed.data.uniform_(-0.1, 0.1)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))

    def forward(self, d, hidden):
        d_embedded = self.embed(d)
        forgetgates, hiddens, cellgates, output = self.lstm(d_embedded, hidden) # hiddens: B * T * hidden_size
        decoded = F.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)), self.embed.weight) 
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens # decoded: B * T * vocab_size


class EncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()
        self.nhid = args.hidden_size
        self.nlayers = args.num_layers

        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.encoder = LSTM(args.embedding_size, args.hidden_size)
        self.decoder = LSTM(args.embedding_size, args.hidden_size)

        self.linear = nn.Linear(self.nhid, args.vocab_size)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

        self.embed.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))

    def forward(self, x, x_mask, y, hidden):
        x_embedded = self.embed(x)
        y_embedded = self.embed(y)

        # encoder
        forgetgates, hiddens, cellgates, output = self.encoder(x_embedded, hidden)
        x_lengths = torch.sum(x_mask, 1).view(x.size(0), 1, 1) - 1
        x_lengths = x_lengths.expand(x.size(0), 1, x_embedded.size(2))
        hiddens = hiddens.gather(1, x_lengths).view(x.size(0), self.nhid)
        cellgates = cellgates.gather(1, x_lengths).view(x.size(0), self.nhid)

        # decoder
        forgetgates, hiddens, cellgates, output = self.decoder(y_embedded, hx=(hiddens, cellgates))

        # output layer
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens


class BiEncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()
        self.nhid = args.hidden_size
        self.nlayers = args.num_layers

        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.fencoder = LSTM(args.embedding_size, args.hidden_size/2)
        self.bencoder = LSTM(args.embedding_size, args.hidden_size/2)
        self.decoder = LSTM(args.embedding_size, args.hidden_size)

        self.linear = nn.Linear(self.nhid, args.vocab_size)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

        self.embed.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid/2).zero_()),
                Variable(weight.new(bsz, self.nhid/2).zero_()))

    def forward(self, x, x_mask, y, hidden):
        x_embedded = self.embed(x)
        y_embedded = self.embed(y)

        # encoder
        forgetgates, hiddens, cellgates, output = self.encoder(x_embedded, hidden)
        x_lengths = torch.sum(x_mask, 1).view(x.size(0), 1, 1) - 1
        x_lengths = x_lengths.expand(x.size(0), 1, x_embedded.size(2))
        hiddens = hiddens.gather(1, x_lengths).view(x.size(0), self.nhid)
        cellgates = cellgates.gather(1, x_lengths).view(x.size(0), self.nhid)

        # decoder
        forgetgates, hiddens, cellgates, output = self.decoder(y_embedded, hx=(hiddens, cellgates))

        # output layer
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens



