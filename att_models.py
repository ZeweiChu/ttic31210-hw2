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

        hx, cx = hx
        all_hx = []
        all_cx = []
        for layer in range(self.num_layers):
            
            weight_ih = getattr(self, "weight_ih_l" + str(layer))
            bias_ih = getattr(self, "bias_ih_l" + str(layer))
            weight_hh = getattr(self, "weight_hh_l" + str(layer))
            bias_hh = getattr(self, "bias_hh_l" + str(layer))
            # code.interact(local=locals())
            gates = F.linear(input, weight_ih, bias_ih) \
                        + F.linear(hx[layer], weight_hh, bias_hh)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cx = (forgetgate * cx[layer]) + (ingate * cellgate)
            hx = outgate * F.tanh(cx)
            all_hx.append(hx)
            all_cx.append(cx)

        return all_hx, all_cx


class ConcatAttentionLayer(nn.Module):
    def __init__(self, args):
        super(ConcatAttentionLayer, self).__init__()
        self.nhid = args.hidden_size
        self.embedding_size = args.embedding_size

        self.linear = nn.Linear(self.nhid + self.nhid, self.nhid)
        self.att_linear = nn.Linear(self.nhid, 1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.att_linear.weight.data.uniform_(-0.1, 0.1)
        self.att_linear.bias.data.zero_()

    def forward(self, s, h):
        B, T, hid_size = s.size()
        s = s.view(B*T, hid_size)
        h = h.view(B*T, hid_size)
        att = torch.cat([s, h], 1)
        att = self.linear(att)
        att = self.att_linear(F.tanh(att)).view(B, T)
        att = torch.exp(F.log_softmax(att))
        return att

class DotAttentionLayer(nn.Module):
    def __init__(self, args):
        super(DotAttentionLayer, self).__init__()
        self.nhid = args.hidden_size
        self.embedding_size = args.embedding_size

    def forward(self, s, h):
        
        att = s*h
        att = torch.sum(att, 2).squeeze(2)
        att = torch.exp(F.log_softmax(att))
        return att


class BilinearAttentionLayer(nn.Module):
    def __init__(self, args):
        super(BilinearAttentionLayer, self).__init__()
        self.nhid = args.hidden_size
        self.embedding_size = args.embedding_size

        self.bilinear = Parameter(torch.Tensor(self.nhid, self.nhid))
        self.bilinear.data.uniform_(-0.1, 0.1)

    def forward(self, s, h):
        B, T, hid_size = s.size()
        s = s.view(B*T, hid_size)
        h = h.view(B*T, hid_size)
        att = F.linear(s, self.bilinear).view(B, T, hid_size)
        att = torch.sum(att * h, 2).squeeze(2)
        att = torch.exp(F.log_softmax(att))
        # code.interact(local=locals())
        return att


'''
    This model is a variation of: 
    Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/abs/1409.0473
'''
class AttentionEncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(AttentionEncoderDecoderModel, self).__init__()
        self.nhid = args.hidden_size
        self.num_layers = args.num_layers

        self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
        self.fencoder = LSTM(args.embedding_size, args.hidden_size/2)
        self.bencoder = LSTM(args.embedding_size, args.hidden_size/2)
        self.decoder = LSTM(args.embedding_size, args.hidden_size)

        if args.attention_type == "dot":
            self.att_layer = DotAttentionLayer(args)
        elif args.attention_type == "bilinear":
            self.att_layer = BilinearAttentionLayer(args)
        elif args.attention_type == "concat":
            self.att_layer = ConcatAttentionLayer(args)


        self.linear = nn.Linear(self.nhid + self.nhid, args.vocab_size)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

        # self.context_linear = nn.Linear(self.nhid + self.nhid, self.nhid)
        # self.context_linear.bias.data.fill_(0)
        # self.context_linear.weight.data.uniform_(-0.1, 0.1)

        self.embed.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        fhx = []
        fcx = []
        bhx = []
        bcx = []
        for i in range(self.num_layers):
            fhx.append(Variable(weight.new(bsz, self.nhid/2).zero_()))
            fcx.append(Variable(weight.new(bsz, self.nhid/2).zero_()))
            bhx.append(Variable(weight.new(bsz, self.nhid/2).zero_()))
            bcx.append(Variable(weight.new(bsz, self.nhid/2).zero_()))

        return ((fhx, fcx), (bhx, bcx))

    def init_decoder_hidden(self, bsz):
        weight = next(self.parameters()).data
        hx = []
        cx = []
        for i in range(self.num_layers):
            hx.append(Variable(weight.new(bsz, self.nhid).zero_()))
            cx.append(Variable(weight.new(bsz, self.nhid).zero_()))

        return (hx, cx)


    def forward(self, x, x_mask, y, hidden):
        x_embedded = self.embed(x)
        y_embedded = self.embed(y)

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
        f_h = hidden[0]
        b_h = hidden[1]
        f_hiddens = []
        b_hiddens = []
        f_cells = []
        b_cells = []
        for i in range(T):
            f_h = self.fencoder(x_embedded[:, i, :], f_h)
            b_h = self.bencoder(x_backward_embedded[:, i, :], b_h)
            f_hiddens.append(f_h[0][-1].unsqueeze(1)) # f_h[0][-1]: hidden state of the last layer
            b_hiddens.append(b_h[0][-1].unsqueeze(1))
            f_cells.append(f_h[1][-1].unsqueeze(1))
            b_cells.append(b_h[1][-1].unsqueeze(1))

        f_hiddens = torch.cat(f_hiddens, 1)
        b_hiddens = torch.cat(b_hiddens, 1)
        f_cells = torch.cat(f_cells, 1)
        b_cells = torch.cat(b_cells, 1)
        hiddens = torch.cat([f_hiddens, b_hiddens], 2)
        cells = torch.cat([f_cells, b_cells], 2)



        # decoder
        B_y, T_y = y.size() 
        h_mean = torch.mean(hiddens, 1).squeeze(1)
        c_mean = torch.mean(cells, 1).squeeze(1)
        hx, cx = [], []
        
        for i in range(self.num_layers):
            hx.append(h_mean.clone())
            cx.append(c_mean.clone())

        y_embedded = self.embed(y)

        context = h_mean
        out_hiddens = []
        
        for i in range(T_y):
            hx, cx = self.decoder(y_embedded[:, i, :], (hx, cx))
            
            att = self.att_layer(hx[-1].unsqueeze(1).expand_as(hiddens).contiguous()\
                , hiddens.contiguous())
            # code.interact(local=locals())
            context = (hiddens * att.unsqueeze(2).expand_as(hiddens)).sum(1).squeeze(1)            
            out_hiddens.append(torch.cat([hx[-1], context], 1).unsqueeze(1))

        out_hiddens = torch.cat(out_hiddens, 1)

        # code.interact(local=locals())
        # output layer
        decoded = self.linear(out_hiddens.view(out_hiddens.size(0)*out_hiddens.size(1), out_hiddens.size(2)))
        decoded = F.log_softmax(decoded)
        return decoded.view(out_hiddens.size(0), out_hiddens.size(1), decoded.size(1)), out_hiddens




