REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent
from .iqn_rnn_agent import IQNRNNAgent
from .riskq_agent_qrdqn import RISKRNNQRAgent
from .card_rnn_agent import CardRNNAgent
from .hdq_card_rnn_agent import HDQCardRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["qrdqn_rnn"] = RISKRNNQRAgent
REGISTRY["card_rnn"] = CardRNNAgent
REGISTRY["hdq_card_rnn"] = HDQCardRNNAgent