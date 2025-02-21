import os

os.environ["WORKING_MODE"] = "train_service"

from gevent import monkey

monkey.patch_all()
from config import global_config

train_config = global_config.train_service_config
os.environ["RWKV_HEAD_SIZE_A"] = str(
            global_config.pretrain_script_config.model.head_size
        )

from RWKV.v7.model import RWKV


model = RWKV(train_config.model)