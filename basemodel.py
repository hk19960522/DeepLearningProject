import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter


class MakeMLP(nn.Module):
    def __init__(self,
                 args, layer_num, layer_name, input_num, hidden_num, output_num,
                 active_fn, dropout, bias, iflastac=True):
        super(MakeMLP, self).__init__()
        layers = []
        self.args = args
        lastac = active_fn if iflastac else ''
        lastdrop = dropout if iflastac else 0

        # set MLP layer
        if layer_num > 1:
            self.addLayer(layers, input_num, hidden_num, bias, active_fn, dropout)
            for i in range(layer_num-2):
                self.addLayer(layers, hidden_num, hidden_num, bias, active_fn, dropout)
            self.addLayer(layers, hidden_num, output_num, bias, lastac, lastdrop)
        else:
            self.addLayer(layers, input_num, output_num, bias, lastac, lastdrop)
        self.MLP = nn.Sequential(*layers)

        layer_name = 'rel'
        # set weights
        if layer_name == 'rel':
            self.MLP.apply(self.init_weights_rel)
        elif layer_name == 'nei':
            self.MLP.apply(self.init_weights_nei)
        elif layer_name == 'attR':
            self.MLP.apply(self.init_weights_attr)
        elif layer_name == 'ngate':
            self.MLP.apply(self.init_weights_ngate)

    def addLayer(self, layers, input_num, output_num, bias, active_fn, dropout):
        layers.append(nn.Linear(input_num, output_num, bias=bias))
        fn = None
        # append activate function
        if active_fn == 'sig':
            fn = nn.Sigmoid
            #fn = torch.sigmoid
        elif active_fn == 'relu':
            fn = nn.ReLU
            #fn = torch.relu
        elif active_fn == 'tanh':
            fn = nn.Tanh
            #fn = torch.tanh
        if fn is not None:
            layers.append(fn())

        # append dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return layers

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            nn.init.constant(m.bias, 0)

    def init_weights_ngate(self, m):
        if type(m) == nn.Linear:
            nn.init.normal(m.weight, std=0.005)
            if self.args.ifbias_gate:
                nn.init.constant(m.bias, 0)

    def init_weights_nei(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal(m.weight, gain=self.args.nei_std)
            if self.args.ifbias_nei:
                nn.init.constant(m.bias, 0)

    def init_weights_attr(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            nn.init.constant(m.bias, 0)

    def init_weights_rel(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=self.args.rela_std)
            if self.args.ifbias_nei:
                nn.init.constant_(m.bias, 0)

class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias=True, num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx, update_mode=''):
        hx, cx = hx
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(cy)

        return outgate, hy, cy


class GCN(nn.Module):
    def __init__(self, args, embed_size, output_size):
        super(GCN, self).__init__()
        self.args = args
        self.relu = nn.ReLU()
        self.embed_size = embed_size
        self.output_size = output_size

        self.dot_size = self.args.hidden_dot_size

        #Motion Gate
        self.ngate = MakeMLP(self.args, 1, 'ngate', self.embed_size + self.output_size * 2, self.args.nei_hidden_size,
                             self.output_size, 'sig', self.args.nei_drop, bias=self.args.ifbias_gate)

        # Relative spatial embedding layer
        self.relativeLayer = MakeMLP(self.args, self.args.rela_layers, 'rel', self.args.rela_input,
                                     self.args.rela_hidden_size,
                                     self.embed_size, self.args.rela_ac, self.args.rela_drop,
                                     bias=True, iflastac=True)
        # Message passing transform
        self.W_nei = MakeMLP(self.args, self.args.nei_layers, 'nei', self.output_size, self.args.nei_hidden_size,
                             self.output_size, self.args.nei_ac, self.args.nei_drop,
                             bias=self.args.ifbias_nei, iflastac=False)

        # Attention
        self.WAr = MakeMLP(self.args, 1, 'arrR', self.embed_size + self.output_size * 2,
                           self.dot_size, 1, self.args.WAr_ac, dropout=0, bias=self.args.ifbias_WAr)

    def forward (self, corr_index, nei_index, nei_num, lstm_state, W):
        '''
        States Refinement process.
        :param corr_index:
            relative coords of each pedestrain pair
        :param nei_index:
            neighbor exsists flag
        :param nei_num:
            neighbor number
        :param lstm_state:
            output states of LSTM Cell
        :param W:
            message passing weight, namely self.W_nei when train one SR Layer
        :return:
            Refined states
            Debug Info
        '''
        outgate, self_h, self_c = lstm_state

        # debug info
        v1, v2, v3 = torch.zeros(1), torch.zeros(1), torch.zeros(1)

        self.N = corr_index.shape[0]

        nei_inputs = self_h.repeat(self.N, 1)

        nei_index_t = nei_index.view((-1))

        corr_t = corr_index.reshape((self.N * self.N, -1))

        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return lstm_state, (0, 0, 0), (0, 0)

        r_t = self.relativeLayer.MLP(corr_t[nei_index_t > 0])
        inputs_part = nei_inputs[nei_index_t > 0]
        hi_t = nei_inputs.view((self.N, self.N, self.output_size)).permute(1, 0, 2).contiguous().view(-1, self.output_size)

        tmp = torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)

        # Motion Gate
        nGate = self.ngate.MLP(tmp)

        # Attention
        Pos_t = torch.full((self.N * self.N, 1), 0, device=torch.device("cuda")).view(-1)
        tt = self.WAr.MLP(torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)).view((-1))
        # have bug if there's any zero value in tt

        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((self.N, self.N))
        Pos[Pos == 0] = -np.Inf
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        # Message Passing
        H = torch.full((self.N * self.N, self.output_size), 0, device=torch.device("cuda"))
        H[nei_index_t > 0] = inputs_part * nGate
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.output_size, 1).transpose(0, 1)
        H = H.view(self.N, self.N, -1)
        H_sum = W.MLP(torch.sum(H, 1))

        # Update Cell states
        C = H_sum + self_c
        H = outgate * F.tanh(C)

        if self.args.ifdebug:
            v1 = torch.norm(H_sum[nei_num > 0] * self.args.nei_ratio) / torch.norm(self_c[nei_num > 0])
            return (outgate, H, C), (v1.item(), v2.item(), v3.item())
        else:
            return (outgate, H, C), (0, 0, 0)


'''
a = MakeMLP(0, 3, 'rel', 5, 128, 256, 'relu', 0.5, 2.0)
print(a)
'''
