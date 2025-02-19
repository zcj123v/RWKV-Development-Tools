import os,json
os.environ["WORKING_MODE"] = "infer_service"

from config import global_config

infer_config = global_config.infer_service_config

from bottle import route, run, request, response,Bottle
import threading
from utils import inference_service_app
from utils.message_manager import Conversation, cList
import torch
# from utils import batching_inference_helper

host = global_config.server_config.infer.host
port = global_config.server_config.infer.port

infer_app=inference_service_app.InferenceAPP()


app=Bottle()


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


def regist_state_id_service():
    req = dict(request.json)
    load_dir = req.get("load_dir", None)  # 从请求中获取 tokens
    return {"access_token": infer_app.regist_state_id(load_dir)}


def remove_state_id_service():
    req = dict(request.json)
    id = req.get("access_token")  # 从请求中获取 tokens
    infer_app.remove_state_id(id)
    return {"message": "success"}


def reset_state_id():
    req = dict(request.json)
    id = req.get("access_token")  # 从请求中获取 tokens
    load_dir = req.get("load_dir", None)  # 从请求中获取 tokens
    infer_app.states_pool[id] = torch.load(load_dir) if load_dir else None
    return {"message": "success"}


def load_weight_service():
    req = dict(request.json)
    load_dir = req.get("load_dir", None)  # 从请求中获取 tokens
    infer_app.model.load_weights(load_dir)
    print(f"load weights from {load_dir}.")
    return {"message": "success"}


def load_state_service():
    req = dict(request.json)
    load_dir = req.get("load_dir")  # 从请求中获取 tokens
    infer_app.load_state(id, load_dir)
    return {"message": "success"}


def save_state_service():
    req = dict(request.json)
    id = req.get("access_token")  # 从请求中获取 tokens
    to_dir = req.get("to_dir", None)  # 从请求中获取 tokens
    infer_app.save_state(id, to_dir)
    return {"message": "success"}


def copy_state_service():
    req = dict(request.json)
    from_id = req.get("from_access_token")  # 从请求中获取 tokens
    to_id = req.get("to_access_token")  # 从请求中获取 tokens
    infer_app.copy_state(from_id, to_id)
    return {"message": "success"}


def infer_batch_service():
    req = dict(request.json)
    tokens_list = req.get("tokens_list")  # 从请求中获取 tokens
    state_idx_list = req.get("state_idx_list", None)  # 可选的 state_idx
    save_cache_dir = req.get("save_cache_dir", None)  # 可选的 state_idx
    state = None
    if state_idx_list:
        for state_idx in state_idx_list:
            if state is None:
                state = infer_app.states_pool[state_idx]
            else:
                state += infer_app.states_pool[state_idx]
    # 调用 infer 函数进行推理
    infer_app.infer_batch(
        tokens_list, state, latent_output=True, save_cache_dir=save_cache_dir
    )
    return {"message": "success"}

def infer_service():
    req = dict(request.json)
    conversations = req.get("conversations")  # 从请求中获取 tokens
    conversations = cList.from_dicts(conversations)
    state_idx = req.get("state_idx", None)  # 可选的 state_idx
    save_logits = req.get("save_logits", True)
    save_folder = req.get("save_folder")
    save_name = req.get("save_name")
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)  # 可选的 state_idx
    state = infer_app.states_pool[state_idx] if state_idx else None
    tokens = conversations.to_tokens(infer_app.tokenizer.encode)[0]
    # 调用 infer 函数进行推理
    (
        logits,
        state,
    ) = infer_app.model.infer(tokens, state)
    if save_to_now_state_idx:
        infer_app.states_pool[save_to_now_state_idx] = state
    if save_logits:
        torch.save(logits.cpu(), os.path.join(save_folder, f"{save_name}.logits"))
    return {"message": "success"}


def infer_tokens_service():
    req = dict(request.json)
    tokens = req.get("tokens")  # 从请求中获取 tokens
    state_idx = req.get("state_idx", None)  # 可选的 state_idx
    save_logits = req.get("save_logits", True)
    save_folder = req.get("save_folder")
    save_name = req.get("save_name")
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)  # 可选的 state_idx
    state = infer_app.states_pool[state_idx] if state_idx else None
    # 调用 infer 函数进行推理
    (
        logits,
        state,
    ) = infer_app.model.infer(tokens, state)
    if save_to_now_state_idx:
        infer_app.states_pool[save_to_now_state_idx] = state
    if save_logits:
        torch.save(logits.cpu(), os.path.join(save_folder, f"{save_name}.logits"))
    # return {
    #     "logits": logits.cpu().float().numpy().tolist(),
    # }
    return {"message": "success"}


def batch_chat_service():
    req = dict(request.json)
    conversations = req.get("conversations")
    resp_start_with_role = req.get("resp_start_with_role")
    resp_start_with_str = req.get("resp_start_with_str")
    stop_with_tokens = req.get("stop_with_tokens")
    stop_supp_tokens = req.get("stop_supp_tokens", [])
    temp = req.get("temp", 1.0)
    top_p = req.get("top_p", 0.7)
    presence_penalty = req.get("presence_penalty", 0.2)
    frequency_penalty = req.get("frequency_penalty", 0.2)
    decay_penalty = req.get("decay_penalty", 0.9961)
    use_now_state_idx = req.get("use_now_state_idx", None)
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)
    max_resp_len = req.get("max_resp_len", 512)
    stream_chunk = req.get("stream_chunk", 9)
    stream = req.get("stream", False)
    format_constrain_str = req.get("format_constrain_str", None)
    token_ban = req.get("token_ban", [])
    init_occurence = req.get("init_occurence", {})

    resp_start_with_tokens = global_config.role[resp_start_with_role][
        "prefix"
    ] + infer_app.tokenizer.encode(resp_start_with_str)
    conversations = cList.from_dicts(conversations) if conversations else None

    if stream:
        return infer_app.batch_chat(
            conversations=conversations,
            resp_start_with_tokens=resp_start_with_tokens,
            stop_with_tokens=stop_with_tokens,
            stop_supp_tokens=stop_supp_tokens,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            decay_penalty=decay_penalty,
            use_now_state_idx=use_now_state_idx,
            save_to_now_state_idx=save_to_now_state_idx,
            max_resp_len=max_resp_len,
            stream_chunk=stream_chunk,
            format_constrain_str=format_constrain_str,
            token_ban=token_ban,
            occurence=init_occurence,
        )
    else:
        resp = ""
        for seg in infer_app.batch_chat(
            conversations=conversations,
            resp_start_with_tokens=resp_start_with_tokens,
            stop_with_tokens=stop_with_tokens,
            stop_supp_tokens=stop_supp_tokens,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            decay_penalty=decay_penalty,
            use_now_state_idx=use_now_state_idx,
            save_to_now_state_idx=save_to_now_state_idx,
            max_resp_len=max_resp_len,
            stream_chunk=stream_chunk,
            format_constrain_str=format_constrain_str,
            token_ban=token_ban,
            occurence=init_occurence,
        ):
            next_txt = seg["next"]
            resp += next_txt
        return resp


def chat_service():
    req = dict(request.json)
    conversations = req.get("conversations")
    resp_start_with_role = req.get("resp_start_with_role")
    resp_start_with_str = req.get("resp_start_with_str")
    stop_with_tokens = req.get("stop_with_tokens")
    stop_supp_tokens = req.get("stop_supp_tokens", [])
    temp = req.get("temp", 1.0)
    top_p = req.get("top_p", 0.7)
    presence_penalty = req.get("presence_penalty", 0.2)
    frequency_penalty = req.get("frequency_penalty", 0.2)
    decay_penalty = req.get("decay_penalty", 0.9961)
    use_now_state_idx = req.get("use_now_state_idx", None)
    save_to_now_state_idx = req.get("save_to_now_state_idx", None)
    chunk_len = req.get("chunk_len", 512)
    max_resp_len = req.get("max_resp_len", 512)
    stream = req.get("stream", False)
    format_constrain_str = req.get("format_constrain_str", None)
    token_ban = req.get("token_ban", [])
    need_ppl = req.get("need_ppl", False)

    resp_start_with_tokens = global_config.role[resp_start_with_role][
        "prefix"
    ] + infer_app.tokenizer.encode(resp_start_with_str)
    conversations = cList.from_dicts(conversations) if conversations else None
    # 设置响应头，流式模式启用分块传输
    if stream:
        return infer_app.chat(
            conversations=conversations,
            resp_start_with_tokens=resp_start_with_tokens,
            stop_with_tokens=stop_with_tokens,
            stop_supp_tokens=stop_supp_tokens,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            decay_penalty=decay_penalty,
            use_now_state_idx=use_now_state_idx,
            save_to_now_state_idx=save_to_now_state_idx,
            chunk_len=chunk_len,
            max_resp_len=max_resp_len,
            stream=stream,
            format_constrain_str=format_constrain_str,
            token_ban=token_ban,
            need_ppl=need_ppl,
        )
    else:
        result_str = infer_app.chat(
            conversations=conversations,
            resp_start_with_tokens=resp_start_with_tokens,
            stop_with_tokens=stop_with_tokens,
            stop_supp_tokens=stop_supp_tokens,
            temp=temp,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            decay_penalty=decay_penalty,
            use_now_state_idx=use_now_state_idx,
            save_to_now_state_idx=save_to_now_state_idx,
            chunk_len=chunk_len,
            max_resp_len=max_resp_len,
            stream=stream,
            format_constrain_str=format_constrain_str,
            token_ban=token_ban,
            need_ppl=need_ppl,
        )
        resp = ""
        for seg in result_str:
            resp += seg

        return resp


def estimate_desire_service():
    req = dict(request.json)
    # 获取请求中的 role 和 prefix
    target_role = req.get("target_role")
    target_prefix = req.get("target_prefix")
    # 获取其他需要的参数
    start_with_tokens = req.get("start_with_tokens")
    ignore_tokens = req.get(
        "ignore_tokens", [11, 33, 261, 263, 41, 42]
    )  # 默认忽略的 tokens
    ignore_tolerance = req.get("ignore_tolerance", 2)
    use_now_state_idx = req.get("use_now_state_idx", None)

    # 获取 target_tokens
    target_tokens = global_config.role[target_role]["prefix"] + infer_app.tokenizer.encode(
        target_prefix
    )

    hit = infer_app.estimate_desires(
        target_tokens=target_tokens,
        start_with_tokens=start_with_tokens,
        ignore_tokens=ignore_tokens,
        ignore_tolerance=ignore_tolerance,
        use_now_state_idx=use_now_state_idx,
    )

    # 返回推断结果
    return {"hit": hit}


app.route("/test", method="GET",callback=test)
app.route("/regist_state_id", method="POST",callback=regist_state_id_service)
app.route("/remove_state_id", method="POST",callback=remove_state_id_service)
app.route("/reset_state_id", method="POST",callback=reset_state_id)
app.route("/load_weight", method="POST",callback=load_weight_service)
app.route("/load_state", method="POST",callback=load_state_service)
app.route("/save_state", method="POST",callback=save_state_service)
app.route("/copy_state", method="POST",callback=copy_state_service)
app.route("/infer_batch", method="POST",callback=infer_batch_service)
app.route("/infer", method="POST",callback=infer_service)
app.route("/infer_tokens", method="POST",callback=infer_tokens_service)
app.route("/chat", method="POST",callback=chat_service)
app.route("/batch_chat", method="POST",callback=batch_chat_service)
app.route("/estimate_desire", method="POST",callback=estimate_desire_service)
import time
def run_server():
    # 启动 Bottle 服务器
    run(app, host=host, port=port)
# def loop():
#     while True:
#         # 使用sleep就有线程运行时间
#         infer_app.batch_helper.loop_step()
#         time.sleep(0.0001)
        
run_server()


# threading.Thread(target=run_server, daemon=True).start()
# threading.Thread(target=loop_step, daemon=True).start()
# loop()