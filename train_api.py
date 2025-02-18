import os
os.environ["WORKING_MODE"] = "train_service"

from gevent import monkey
monkey.patch_all()
from config import global_config

train_config = global_config.train_service_config


from bottle import route, run, request, response
from utils.message_manager import cList, Conversation
import torch.distributed as dist
import requests
import asyncio
import aiohttp
from utils.train_app import OnlineTrainingAPP
import json

app = OnlineTrainingAPP()
rank = dist.get_rank()
total_ranks = dist.get_world_size()
port = global_config.server_config.train.port_begin + rank


@route("/test", method="GET")
def test():
    try:
        # 设置响应为JSON格式
        response.content_type = "application/json"
        # 返回服务器状态
        return json.dumps({"status": "running", "code": 200})
    except Exception as e:
        # 捕捉异常并返回错误信息
        response.status = 500  # Internal Server Error
        return json.dumps({"status": "error", "message": str(e)})


@route("/save", method="POST")
def save():
    req = dict(request.json)
    model_name = req.get("save_name", "default")
    save_train_state = req.get("save_train_state", False)
    folder = req.get("folder", None)
    app.save_weight(model_name, save_train_state, folder=folder)
    return {"message": "success"}


@route("/load_model", method="POST")
def load_model():
    req = dict(request.json)
    ckpt_dir = req.get("load_dir", "")
    app.load_model(ckpt_dir)
    return {"message": "success"}


@route("/train_single_datafolder", method="POST")
def train_single_datafolder():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_single_datafolder"))

    forder_dir = req.get("forder_dir")
    epoch = req.get("epoch", 1)
    batch_size_per_gpu = req.get("batch_size_per_gpu", 1)
    n_save_ckpt = req.get("n_save_ckpt", 1)
    multi_scale_ctx = req.get("multi_scale_ctx", train_config.model.ctx_len)
    multi_scale_alpha = req.get(
        "multi_scale_alpha", train_config.train.multi_scale_alpha
    )
    min_loss = req.get("min_loss", train_config.train.min_loss)
    max_loss = req.get("max_loss", train_config.train.max_loss)
    min_loss_fix = req.get("min_loss_fix", train_config.train.min_loss_fix)
    max_loss_fix = req.get("max_loss_fix", train_config.train.max_loss_fix)
    n_save_step = req.get("n_save_step", None)
    keep_states_mode = req.get("keep_states_mode", "never")
    dataloader_workers_per_gpu = req.get("dataloader_workers_per_gpu", 4)
    begin_with_state_dir=req.get("begin_with_state_dir",None)
    use_qa_mask = req.get("use_qa_mask", False)
    

    response.content_type = "application/json"
    return app.train_from_folder(
        forder_dir=forder_dir,
        epoch=epoch,
        batch_size_per_gpu=batch_size_per_gpu,
        n_save_ckpt=n_save_ckpt,
        multi_scale_ctx=multi_scale_ctx,
        multi_scale_alpha=multi_scale_alpha,
        min_loss=min_loss,
        max_loss=max_loss,
        min_loss_fix=min_loss_fix,
        max_loss_fix=max_loss_fix,
        n_save_step=n_save_step,
        keep_states_mode=keep_states_mode,
        dataloader_workers_per_gpu=dataloader_workers_per_gpu,
        begin_with_state_dir=begin_with_state_dir,
        use_qa_mask=use_qa_mask,
    )
    
   
        
        

@route("/train_from_folders", method="POST")
def train_from_folders():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_text_from_folders"))
    folder_weight_dir_list = req.get("folder_weight_dir_list", [])
    epoch = req.get("epoch", 1)
    batch_size_per_gpu = req.get("batch_size_per_gpu", 1)
    n_save_ckpt = req.get("n_save_ckpt", 1)
    min_loss = req.get("min_loss", train_config.train.min_loss)
    max_loss = req.get("max_loss", train_config.train.max_loss)
    min_loss_fix = req.get("min_loss_fix", train_config.train.min_loss_fix)
    max_loss_fix = req.get("max_loss_fix", train_config.train.max_loss_fix)
    n_save_step = req.get("n_save_step", None)
    dataloader_workers_per_gpu = req.get("dataloader_workers_per_gpu", 2)
    use_qa_mask = req.get("use_qa_mask", False)
    
    response.content_type = "application/json"
    return app.train_from_folders(
        folder_weight_dir_list=folder_weight_dir_list,
        epoch=epoch,
        batch_size_per_gpu=batch_size_per_gpu,
        n_save_ckpt=n_save_ckpt,
        min_loss=min_loss,
        max_loss=max_loss,
        min_loss_fix=min_loss_fix,
        max_loss_fix=max_loss_fix,
        n_save_step=n_save_step,
        dataloader_workers_per_gpu=dataloader_workers_per_gpu,
        use_qa_mask=use_qa_mask,
    )

        
@route("/train_text_from_messages", method="POST")
def train_text_from_messages():
    req = dict(request.json)
    if rank == 0 and total_ranks > 1:
        asyncio.run(distribute_package(req, "train_text_from_messages"))
    messages = req.get("messages", [])
    messages = [cList.from_dicts(msgs) for msgs in messages]
    batch_size = req.get("batch_size", 1)
    n_save_ckpt = req.get("n_save_ckpt", -1)
    min_loss = req.get("min_loss", train_config.train.min_loss)
    max_loss = req.get("max_loss", train_config.train.max_loss)
    min_loss_fix = req.get("min_loss_fix", train_config.train.min_loss_fix)
    max_loss_fix = req.get("max_loss_fix", train_config.train.max_loss_fix)
    multi_scale_ctx = req.get("multi_scale_ctx", train_config.model.ctx_len)
    multi_scale_alpha = req.get(
        "multi_scale_alpha", train_config.train.multi_scale_alpha
    )
    keep_train_states = req.get("keep_train_states", False)
    use_ego_mask = req.get("use_ego_mask", False)
    ignore_ctx = req.get("ignore_ctx", False)
    save_name_last = req.get("save_name_last", "last")
    app.train_text_from_messages(
        messages=messages,
        batch_size=batch_size,
        n_save_ckpt=n_save_ckpt,
        min_loss=min_loss,
        max_loss=max_loss,
        min_loss_fix=min_loss_fix,
        max_loss_fix=max_loss_fix,
        multi_scale_ctx=multi_scale_ctx,
        multi_scale_alpha=multi_scale_alpha,
        keep_train_states=keep_train_states,
        use_ego_mask=use_ego_mask,
        ignore_ctx=ignore_ctx,
    )
    app.save_weight(save_name_last, True)



async def distribute_package(received_package, route):
    async def send_to_rank(r):
        other_rank_port = global_config.server_config.train.port_begin + r
        url = f"http://localhost:{other_rank_port}/{route}"
        try:
            print(f"distribute package from {rank} to rank {r}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=received_package) as response:
                    await response.text()
        except Exception as e:
            print(f"Error sending package to rank {r}: {e}")

    tasks = []
    for r in range(total_ranks):
        if r != rank:  # 不需要将数据包发送到自己
            tasks.append(send_to_rank(r))

    await asyncio.gather(*tasks)


run(host="0.0.0.0", port=port)
