REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC
from .base_controller import BaseCentralMAC
from .ResZ_controller import ResZ_controller
from .sdq_controller import SDQ
from .risk_controller import RiskMAC
from .rlcard_controller import RlCardMAC
from .hdq_rlcard_controller import  HDQRlCardMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY['resz_mac'] = ResZ_controller
REGISTRY['base_central_mac'] = BaseCentralMAC
REGISTRY['sdq'] = SDQ
REGISTRY['risk_mac'] = RiskMAC
REGISTRY['rlcard'] = RlCardMAC
REGISTRY['rlcard_hdq'] = HDQRlCardMAC