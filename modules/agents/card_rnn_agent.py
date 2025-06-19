import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class CardRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CardRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape+54, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        b,  e = inputs.size()


        x = F.relu(self.fc1(inputs), inplace=True)
        # x = self.fc1(inputs).pow(2)
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # hh = self.rnn(x, h_in)
        x = F.relu(self.fc3(x), inplace=True)
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(x))
        else:
            q = self.fc2(x)

        return q.view(b,  -1)