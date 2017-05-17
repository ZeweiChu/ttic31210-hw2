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
        decoded = F.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)), self.out_embed) 
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens # decoded: B * T * vocab_size

class LSTMHingeOutEmbNegModel(nn.Module):
    def __init__(self, args):
        super(LSTMHingeOutEmbNegModel, self).__init__()
        self.nhid = args.hidden_size
        self.nlayers = args.num_layers
        self.num_sampled = args.num_sampled
        self.vocab_size = args.vocab_size

        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.lstm = LSTM(args.embedding_size, args.hidden_size)
        self.out_embed = nn.Embedding(args.vocab_size, args.hidden_size)
        
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.out_embed.weight.data.uniform_(-0.1, 0.1)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))

    def forward(self, d, hidden, output):
        B, T = d.size()
        d_embedded = self.embed(d)
        # code.interact(local=locals())
        forgetgates, hiddens, cellgates, _ = self.lstm(d_embedded, hidden) # hiddens: B * T * hidden_size
        
        # negative sampling
        noise = Variable(torch.Tensor(B * T, self.num_sampled).uniform_(0, self.vocab_size-1).long())
        noise = self.out_embed(noise) # (B * T) * num_sampled * nhid

        decoded = torch.bmm(noise, hiddens.view(B*T, self.nhid, 1)).squeeze(2) # (B * T) * num_sampled

        out_embed = self.out_embed(output)
        out = torch.bmm(out_embed.view(B*T, 1, self.nhid), hiddens.view(B*T, self.nhid, 1)).squeeze(2)
        decoded = torch.cat([out, decoded], 1)

        return decoded.view(B, T, self.num_sampled+1), hiddens # decoded: B * T * vocab_size

    def predict(self, d, hidden):
        B, T = d.size()
        d_embedded = self.embed(d)
        forgetgates, hiddens, cellgates, output = self.lstm(d_embedded, hidden) # hiddens: B * T * hidden_size
        
        # negative sampling
        # code.interact(local=locals())
        noise = Variable(torch.arange(0, self.vocab_size).long()).unsqueeze(0).expand(B*T, self.vocab_size)
        noise = self.out_embed(noise) # (B * T) * num_sampled * nhid
        # code.interact(local=locals())
        decoded = torch.bmm(noise, hiddens.view(B*T, self.nhid, 1)).squeeze(2) # (B * T) * num_sampled
        return decoded.view(B, T, self.vocab_size), hiddens # decoded: B * T * vocab_size


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

    def encode(self, x, x_mask, hidden):
        x_embedded = self.embed(x)
        forgetgates, hiddens, cellgates, output = self.encoder(x_embedded, hidden)
        x_lengths = torch.sum(x_mask, 1).view(x.size(0), 1, 1) - 1
        x_lengths = x_lengths.expand(x.size(0), 1, hiddens.size(2))
        hiddens = hiddens.gather(1, x_lengths).view(x.size(0), self.nhid)
        cellgates = cellgates.gather(1, x_lengths).view(x.size(0), self.nhid)

        return hiddens, cellgates

    def forward(self, x, x_mask, y, hidden):
        
        # encoder
        hiddens, cellgates = self.encode(x, x_mask, hidden)
        y_embedded = self.embed(y)

        # decoder
        forgetgates, hiddens, cellgates, output = self.decoder(y_embedded, hx=(hiddens, cellgates))

        # output layer
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens


class BiEncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(BiEncoderDecoderModel, self).__init__()
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
                Variable(weight.new(bsz, self.nhid/2).zero_())), (Variable(weight.new(bsz, self.nhid/2).zero_()),
                Variable(weight.new(bsz, self.nhid/2).zero_()))

    def encode(self, x, x_mask, hidden):
        x_embedded = self.embed(x)

        B, T = x.size()
        rev_index = torch.arange(T-1, -1, -1).view(1,-1).expand(B, T).long()
        mask_length = torch.sum(1-x_mask.data, 1).long().expand_as(rev_index)
        rev_index -= mask_length
        rev_index[rev_index < 0] = 0
        rev_index = Variable(rev_index)

        x_backward = Variable(x.data.new(x.data.size()).fill_(0))
        x_backward.scatter_(1, rev_index, x)
        x_backward_embedded = self.embed(x_backward)

        # encoder
        f_forgetgates, f_hiddens, f_cellgates, f_output = self.fencoder(x_embedded, hidden[0])
        b_forgetgates, b_hiddens, b_cellgates, b_output = self.bencoder(x_backward_embedded, hidden[1])
        x_lengths = torch.sum(x_mask, 1).view(x.size(0), 1, 1) - 1
        x_lengths = x_lengths.expand(x.size(0), 1, f_hiddens.size(2))
        # code.interact(local=locals())
        f_hiddens = f_hiddens.gather(1, x_lengths).view(x.size(0), self.nhid/2)
        f_cellgates = f_cellgates.gather(1, x_lengths).view(x.size(0), self.nhid/2)
        b_hiddens = b_hiddens.gather(1, x_lengths).view(x.size(0), self.nhid/2)
        b_cellgates = b_cellgates.gather(1, x_lengths).view(x.size(0), self.nhid/2)

        hiddens = torch.cat([f_hiddens, b_hiddens], 1)
        cellgates = torch.cat([f_cellgates, b_cellgates], 1)
        return hiddens, cellgates

    def forward(self, x, x_mask, y, hidden):
        y_embedded = self.embed(y)
        hiddens, cellgates = self.encode(x, x_mask, hidden)

        # decoder
        forgetgates, hiddens, cellgates, output = self.decoder(y_embedded, hx=(hiddens, cellgates))

        # output layer
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens


class BOWEncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(BOWEncoderDecoderModel, self).__init__()
        self.embedding_size = args.embedding_size
        self.nhid = args.hidden_size
        self.vocab_size = args.vocab_size

        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.h_linear = nn.Linear(self.embedding_size, self.nhid)
        self.c_linear = nn.Linear(self.embedding_size, self.nhid)
        self.decoder = LSTM(args.embedding_size, args.hidden_size)
        self.linear = nn.Linear(self.nhid, args.vocab_size) # output decoding layer
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)


        self.h_linear.bias.data.fill_(0)
        self.h_linear.weight.data.uniform_(-0.1, 0.1)
        self.c_linear.bias.data.fill_(0)
        self.c_linear.weight.data.uniform_(-0.1, 0.1)

        self.embed.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))

    def encode(self, x, x_mask, hidden):
        x_embedded = self.embed(x)
        
        mask_3d = x_mask.unsqueeze(2).expand_as(x_embedded)
        x_embedded[mask_3d == 0] = 0.
        x_avg = torch.sum(x_embedded, 1).squeeze(1) / torch.sum(mask_3d, 1).squeeze(1).float()
        
        hidden = self.h_linear(x_avg)
        cell = self.c_linear(x_avg)
        return hidden, cell

    def forward(self, x, x_mask, y, hidden):
        y_embedded = self.embed(y)
        hidden, cell = self.encode(x, x_mask, hidden)
        
        #decoder
        forgetgates, hiddens, cellgates, output = self.decoder(y_embedded, hx=(hidden, cell))
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens




