import os

os.environ["WORKING_MODE"] = "train_service"

from gevent import monkey

monkey.patch_all()
from config import global_config

train_config = global_config.train_service_config

from RWKV.v7.model import RWKV


