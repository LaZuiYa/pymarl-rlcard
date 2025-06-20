from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .macf import MultiAgentCar

from rlcard.envs.doudizhu import DoudizhuEnv
try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["macf"] = partial(env_fn, env=MultiAgentCar)


REGISTRY["doudizhu"] = partial(env_fn, env=DoudizhuEnv) #smacv2
if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
