from config import global_config
import os

infer_config = global_config.infer_service_config

from utils.message_manager import Conversation, cList
from utils.collections import parse_format_constrain_str
from config import RWKVInfer as RWKV
from typing import List
from RWKV.functions import sample_logits, ppl
from config import BlockStateList
import torch
import sys, gc
import uuid
import json
import re
from utils import batching_inference_helper
import socket
import time


class InferenceAPP:
    def __init__(self):
        gc.collect()
        self.tokenizer = global_config.tokenizer_eval
        self.model = RWKV(infer_config)
        self.states_pool = {}
        self.res_buffer = {}
        self.broadcast_host = global_config.server_config.infer.batching_broadcast_host
        self.broadcast_port = global_config.server_config.infer.batching_broadcast_port
        self.batch_helper = batching_inference_helper.BatchingInferenceHelper(
            self.batch_block_infer,
            max_bsz=5,
            broadcast_host=self.broadcast_host,
            broadcast_port=self.broadcast_port,
        )

    def batch_block_infer(
        self, tokens_batches: list, state: BlockStateList, chunk_len: int = 512
    ):
        out = None
        t_batches = [x[:chunk_len] for x in tokens_batches]
        last_len = len(t_batches[0])
        assert all(len(x) == last_len for x in t_batches)
        while last_len > 0:
            out, state = self.model.batching(t_batches, state)
            tokens_batches = [x[chunk_len:] for x in tokens_batches]
            t_batches = [x[:chunk_len] for x in tokens_batches]
            last_len = len(t_batches[0])
        return out, state

    def regist_state_id(self, load_dir: str = None):
        state_id = "sk-" + str(uuid.uuid4())
        self.states_pool[state_id] = (
            torch.load(load_dir, map_location=next(self.model.parameters()).device)
            if load_dir
            else None
        )
        return state_id

    def save_state(self, state_id: str, to_dir: str):
        if state_id in self.states_pool and self.states_pool[state_id]:
            torch.save(self.states_pool[state_id], to_dir)

    def load_state(self, state_id: str, load_dir: str = None):
        try:
            self.states_pool[state_id] = torch.load(load_dir) if load_dir else None
        except:
            print(f"load state: state not found")
            self.states_pool[state_id]
        print(f"load state id={state_id} from {load_dir}.")

    def remove_state_id(self, state_id: str):
        if state_id in self.states_pool.keys():
            del self.states_pool[state_id]

    def copy_state(self, state_id: str, target_state_id: str):
        if state_id in self.states_pool.keys():
            self.states_pool[target_state_id] = self.states_pool[state_id]

    def infer_batch(
        self, tokens_batches: list, state: BlockStateList = None, latent_output=False
    ):
        if latent_output:
            out, state, latent_out = self.model.batching(
                tokens_batches, state, latent_output
            )
            return out, state, latent_out
        out, state = self.model.batching(tokens_batches, state)
        return out, state

    def block_infer(self, tokens, state, chunk_len=512):
        out = None
        tokens = [int(x) for x in tokens]
        # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')
        while len(tokens) > 0:
            out, state = self.model(tokens[:chunk_len], state)
            tokens = tokens[chunk_len:]
        return out, state

    def chat(
        self,
        conversations: cList,
        resp_start_with_tokens: List[int],
        stop_with_tokens: List[int],
        stop_supp_tokens: List[int],
        temp: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        decay_penalty: float,
        use_now_state_idx=None,
        save_to_now_state_idx=None,
        chunk_len: int = 512,
        max_resp_len: int = 512,
        stream=False,
        stream_chunk=9,
        format_constrain_str=None,
        token_out=False,
        token_ban=[],
        need_ppl=False,
    ):
        in_ppls = []
        state = self.states_pool[use_now_state_idx] if use_now_state_idx else None
        upload_tokens = (
            conversations.to_tokens(self.tokenizer.encode)[0] + resp_start_with_tokens
            if conversations
            else resp_start_with_tokens if resp_start_with_tokens else [0]
        )
        out, new_state = self.block_infer(
            upload_tokens,
            state,
            chunk_len,
        )
        if len(upload_tokens) > 1:
            in_ppls += ppl(upload_tokens[1:], out[0, :-1, :])
        occurrence = {}
        tokens = []
        start_index = 0
        count = 0
        out_ppls = []
        if format_constrain_str is not None:
            blocks = parse_format_constrain_str(format_constrain_str)
            for block in blocks:
                if block["type"] == "str":
                    cl = cList()
                    txt = block["text"]

                    c = Conversation(role="text", content=txt)
                    cl.append(c)
                    out, new_state = self.block_infer(
                        cl.to_tokens(self.tokenizer.encode)[0],
                        new_state,
                        chunk_len,
                    )
                    if need_ppl:
                        out_ppls += ppl(
                            cl.to_tokens(self.tokenizer.encode)[0][1:], out[0, :-1, :]
                        )
                    print(txt, end="")
                    sys.stdout.flush()
                    if save_to_now_state_idx:
                        self.states_pool[save_to_now_state_idx] = new_state
                    if token_out:
                        pass
                    elif stream:
                        package = {"next": txt}
                        if need_ppl:
                            package["in_ppls"] = in_ppls
                            package["out_ppls"] = out_ppls
                        yield json.dumps(package, ensure_ascii=True) + "\n"
                    else:
                        if need_ppl:
                            yield txt, in_ppls, out_ppls
                        else:
                            yield txt
                elif block["type"] == "select":
                    selections = block["selections"]
                    select_tokens = [
                        self.tokenizer.encode(s.strip())
                        for s in selections
                        if s.strip()
                    ]
                    while select_tokens:
                        allow_tokens = [s[0] for s in select_tokens]
                        for n in occurrence:
                            out[0, -1, n] -= (
                                presence_penalty + occurrence[n] * frequency_penalty
                            )
                        if allow_tokens:
                            mask = torch.full_like(
                                out[0, -1, :], -1e38, device=out.device
                            )
                            mask[allow_tokens] = out[0, -1, allow_tokens]
                            out[0, -1, :] = mask
                        for tttt in token_ban:
                            out[0, -1, tttt] = torch.full_like(
                                out[0, -1, tttt], -1e38, device=out.device
                            )
                        token = sample_logits(
                            out[0, -1, :],
                            temperature=temp,
                            top_p=top_p,
                        )
                        if need_ppl:
                            token_ppl = ppl([token], out[0, -1:, :])
                            out_ppls += token_ppl
                        for xxx in occurrence:
                            occurrence[xxx] *= decay_penalty
                        if 49 <= token <= 58:
                            pass
                        elif token not in occurrence:
                            occurrence[token] = 1
                        else:
                            occurrence[token] += 1
                        select_tokens = [
                            s[1:]
                            for s in select_tokens
                            if s and token == s[0] and s[1:]
                        ]
                        # print("select->" ,select_tokens)
                        out, new_state = self.model([token], new_state)
                        tokens += [token]
                        if token_out:
                            yield token
                        count += 1
                        if count > stream_chunk and "�" not in self.tokenizer.decode(
                            tokens[-1:]
                        ):
                            count = 0
                            txt = self.tokenizer.decode(tokens[start_index:])
                            start_index = len(tokens)
                            if token_out:
                                pass
                            elif stream:
                                package = {"next": txt}
                                if need_ppl:
                                    package["in_ppls"] = in_ppls
                                    package["out_ppls"] = out_ppls
                                yield json.dumps(package, ensure_ascii=True) + "\n"
                            # else:
                            print(txt, end="")
                            sys.stdout.flush()
                    if count > 0:
                        count = 0
                        txt = self.tokenizer.decode(tokens[start_index:])
                        start_index = len(tokens)
                        if token_out:
                            pass
                        elif stream:
                            package = {"next": txt}
                            if need_ppl:
                                package["in_ppls"] = in_ppls
                                package["out_ppls"] = out_ppls
                            yield json.dumps(package, ensure_ascii=True) + "\n"
                        # else:
                        print(txt, end="")
                        sys.stdout.flush()
                    if save_to_now_state_idx:
                        self.states_pool[save_to_now_state_idx] = new_state
                    if not stream:
                        send_msg = self.tokenizer.decode(tokens)
                        tokens = []
                        if need_ppl:
                            yield send_msg, in_ppls, out_ppls
                        elif not token_out:
                            yield send_msg
                elif block["type"] == "generate":
                    max_len = block["n_max"]
                    stop = block["stop"]
                    supplement = block["supplement"]
                    for i in range(max_len):
                        for n in occurrence:
                            out[0, -1, n] -= (
                                presence_penalty + occurrence[n] * frequency_penalty
                            )
                        for tttt in token_ban:
                            out[0, -1, tttt] = torch.full_like(
                                out[0, -1, tttt], -1e38, device=out.device
                            )

                        token = sample_logits(
                            out[0, -1, :],
                            temperature=temp,
                            top_p=top_p,
                        )
                        if need_ppl:
                            token_ppl = ppl([token], out[0, -1:, :])
                            out_ppls += token_ppl
                        for xxx in occurrence:
                            occurrence[xxx] *= decay_penalty
                        if 49 <= token <= 58:
                            pass
                        elif token not in occurrence:
                            occurrence[token] = 1
                        else:
                            occurrence[token] += 1
                        out, new_state = self.model([token], new_state)
                        tokens += [token]
                        count += 1
                        if token in stop or i == max_len - 1:  # 结束
                            if supplement:
                                out, new_state = self.model(supplement, new_state)
                                tokens.pop()
                                count -= 1
                            if count > 0:
                                count = 0
                                txt = self.tokenizer.decode(tokens[start_index:])
                                start_index = len(tokens)
                                if token_out:
                                    pass
                                elif stream:
                                    package = {"next": txt}
                                    if need_ppl:
                                        package["in_ppls"] = in_ppls
                                        package["out_ppls"] = out_ppls
                                    yield json.dumps(package, ensure_ascii=True) + "\n"
                                # else:
                                print(txt, end="")
                                sys.stdout.flush()
                            break
                        if token_out:
                            yield token
                        if count > stream_chunk and "�" not in self.tokenizer.decode(
                            tokens[-1:]
                        ):
                            count = 0
                            txt = self.tokenizer.decode(tokens[start_index:])
                            start_index = len(tokens)
                            if token_out:
                                pass
                            elif stream:
                                package = {"next": txt}
                                if need_ppl:
                                    package["in_ppls"] = in_ppls
                                    package["out_ppls"] = out_ppls
                                yield json.dumps(package, ensure_ascii=True) + "\n"
                            # else:
                            print(txt, end="")
                            sys.stdout.flush()
                    if save_to_now_state_idx:
                        self.states_pool[save_to_now_state_idx] = new_state
                    if not stream:
                        send_msg = self.tokenizer.decode(tokens)
                        tokens = []
                        if need_ppl:
                            yield send_msg, in_ppls, out_ppls
                        elif not token_out:
                            yield send_msg
        else:
            for i in range(max_resp_len):
                for n in occurrence:
                    out[0, -1, n] -= (
                        presence_penalty + occurrence[n] * frequency_penalty
                    )
                for tttt in token_ban:
                    out[0, -1, tttt] = torch.full_like(
                        out[0, -1, tttt], -1e38, device=out.device
                    )

                token = sample_logits(
                    out[0, -1, :],
                    temperature=temp,
                    top_p=top_p,
                )
                if need_ppl:
                    token_ppl = ppl([token], out[0, -1:, :])
                    out_ppls += token_ppl
                for xxx in occurrence:
                    occurrence[xxx] *= decay_penalty
                if 49 <= token <= 58:
                    pass
                elif token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1
                out, new_state = self.model([token], new_state)
                tokens += [token]
                count += 1
                if token in stop_with_tokens or i == max_resp_len - 1:  # 结束
                    tokens.pop()
                    count -= 1
                    if stop_supp_tokens:
                        out, new_state = self.model(stop_supp_tokens, new_state)
                    if count > 0:
                        count = 0
                        txt = self.tokenizer.decode(tokens[start_index:])
                        start_index = len(tokens)
                        if token_out:
                            pass
                        elif stream:
                            package = {"next": txt}
                            if need_ppl:
                                package["in_ppls"] = in_ppls
                                package["out_ppls"] = out_ppls
                            yield json.dumps(package, ensure_ascii=True) + "\n"
                        # else:
                        print(txt, end="")
                        sys.stdout.flush()
                    break
                if token_out:
                    yield token
                if count > stream_chunk and "�" not in self.tokenizer.decode(
                    tokens[-1:]
                ):
                    count = 0
                    txt = self.tokenizer.decode(tokens[start_index:])
                    start_index = len(tokens)
                    if token_out:
                        pass
                    elif stream:
                        package = {"next": txt}
                        if need_ppl:
                            package["in_ppls"] = in_ppls
                            package["out_ppls"] = out_ppls
                        yield json.dumps(package, ensure_ascii=True) + "\n"
                    # else:
                    print(txt, end="")
                    sys.stdout.flush()
            if save_to_now_state_idx:
                self.states_pool[save_to_now_state_idx] = new_state
            if not stream:
                send_msg = self.tokenizer.decode(tokens)
                tokens = []
                if need_ppl:
                    yield send_msg, out_ppls
                elif not token_out:
                    yield send_msg

    def batch_chat(
        self,
        conversations: cList,
        resp_start_with_tokens: List[int],
        stop_with_tokens: List[int],
        stop_supp_tokens: List[int],
        temp: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        decay_penalty: float,
        use_now_state_idx=None,
        save_to_now_state_idx=None,
        max_resp_len: int = 512,
        stream_chunk=9,
        format_constrain_str=None,
        token_ban=[],
        occurence={},
    ):
        if use_now_state_idx is None:
            user_id = self.regist_state_id()
        else:
            user_id = use_now_state_idx

        self.batch_helper.add_task(
            user_id,
            self.states_pool[user_id],
            (
                conversations.to_tokens(self.tokenizer.encode)[0]
                + resp_start_with_tokens
                if conversations
                else resp_start_with_tokens if resp_start_with_tokens else [0]
            ),
            temp,
            top_p,
            presence_penalty,
            frequency_penalty,
            decay_penalty,
            format_constrain_str,
            stop_with_tokens,
            stop_supp_tokens,
            max_resp_len,
            token_ban,
            occurence,
        )
        stream_tokens = []
        count = 0
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(
            (self.broadcast_host, self.broadcast_port)
        )  

        while True:
            time.sleep(0.0001)
            message, client_address = server_socket.recvfrom(4096)
            message_str = message.decode()
            message_json = json.loads(message_str)
            if user_id in message_json:
                return_dict = message_json[user_id]
                # resp_tokens=return_dict["resp_tokens"]
                # iter_tokens=return_dict["iter_tokens"]
                # state=return_dict["state"]
                # in_ppls=return_dict["in_ppls"]
                # out_ppls=return_dict["out_ppls"]
                # occurence=return_dict["occurence"]
                over = return_dict["over"]
                success = return_dict["success"]

                matching_task = next(
                    (
                        task
                        for task in self.batch_helper.batch_tasks
                        if task.user_id == user_id
                    ),
                    None,
                )
                if not success:
                    break
                if over:
                    if stream_tokens:
                        next_texts = self.tokenizer.decode(stream_tokens)
                        stream_tokens.clear()
                        if save_to_now_state_idx:
                            self.states_pool[save_to_now_state_idx] = (
                                matching_task.state
                            )

                        in_ppls = matching_task.in_ppls
                        out_ppls = matching_task.out_ppls
                        
                        self.batch_helper.batch_tasks.remove(matching_task)
                        package = {
                            "next": next_texts,
                            "in_ppls": in_ppls,
                            "out_ppls": out_ppls,
                        }
                        yield json.dumps(package, ensure_ascii=True) + "\n"
                        
                    break
                if matching_task is not None:
                    stream_tokens += matching_task.iter_tokens
                    if len(stream_tokens) > stream_chunk:
                        next_texts = self.tokenizer.decode(stream_tokens)
                        stream_tokens.clear()
                        if save_to_now_state_idx:
                            self.states_pool[save_to_now_state_idx] = (
                                matching_task.state
                            )

                        in_ppls = matching_task.in_ppls
                        out_ppls = matching_task.out_ppls
                        

                        package = {
                            "next": next_texts,
                            "in_ppls": in_ppls,
                            "out_ppls": out_ppls,
                        }
                        yield json.dumps(package, ensure_ascii=True) + "\n"



    # def udp_server():
    #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     server_socket.bind(("0.0.0.0", 12345))  # 监听所有网卡的 12345 端口

    #     print("UDP server listening on port 12345...")

    #     while True:
    #         message, client_address = server_socket.recvfrom(1024)  # 1024 字节为最大接收数据量
    #         print(f"Received message: {message.decode()} from {client_address}")

    # if __name__ == "__main__":
    #     udp_server()

    def estimate_desires(
        self,
        target_tokens,
        start_with_tokens,
        ignore_tokens=[11, 33, 261, 263, 41, 42],
        ignore_tolerance=2,
        use_now_state_idx=None,
    ):
        max_len = len(target_tokens) + ignore_tolerance
        resp_tokens = []
        for t in self.chat(
            None,
            start_with_tokens,
            [],
            [],
            0.2,
            1,
            0,
            0,
            1,
            use_now_state_idx,
            max_resp_len=max_len,
            token_out=True,
        ):
            resp_tokens.append(t)
        for t in ignore_tokens:
            while t in resp_tokens:
                resp_tokens.remove(t)
            while t in target_tokens:
                target_tokens.remove(t)

        print(f"{resp_tokens}←→{target_tokens}")

        return resp_tokens[: len(target_tokens)] == target_tokens
