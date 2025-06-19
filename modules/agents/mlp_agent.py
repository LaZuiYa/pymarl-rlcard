import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim, bias=False)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions, bias=False)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def init_hidden(self):
        return None

    def forward(self, inputs, h=None):
        x = self.fc1(inputs).pow(2)
        x = self.fc2(x).pow(2)
        # if self.agent_return_logits:
        #     actions = self.fc3(x)
        # else:
        #     actions = F.tanh(self.fc3(x))
        actions = self.fc3(x)
        return actions