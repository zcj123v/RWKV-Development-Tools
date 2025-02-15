from config import global_config

infer_config = global_config.infer_service_config

from RWKV.functions import sample_logits, ppl
from config import BlockStateList
import torch
from utils.collections import parse_format_constrain_str
from collections import OrderedDict
import requests
import asyncio
from typing import List
import socket
import json


class ConstraintGenerateBlockList:
    def __init__(self, task, constraint_str):
        self.belonging_task = task
        if constraint_str is None:
            self.constraint_blocks = [
                ConstraintGenerateBlock(
                    "generate", max_tokens=task.max_tokens, token_stop=task.token_stop
                )
            ]
        else:
            try:
                str_blocks = parse_format_constrain_str(constraint_str)
                self.constraint_blocks = []
                for block in str_blocks:
                    if block["type"] == "str":
                        self.constraint_blocks.append(
                            ConstraintGenerateBlock("str", text=block["text"])
                        )
                    elif block["type"] == "select":
                        self.constraint_blocks.append(
                            ConstraintGenerateBlock(
                                "select", selections=block["select"]
                            )
                        )
                    elif block["type"] == "generate":
                        self.constraint_blocks.append(
                            ConstraintGenerateBlock(
                                "generate",
                                max_tokens=block["n_max"],
                                token_stop=block["stop"],
                            )
                        )
            except Exception as e:
                print(
                    f"编号{task.user_id}的任务解析约束字符串失败，使用默认生成方式。错误信息：{e}"
                )
                self.constraint_blocks = [
                    ConstraintGenerateBlock(
                        "generate",
                        max_tokens=task.max_tokens,
                        token_stop=task.token_stop,
                    )
                ]
        self.now_block = self.constraint_blocks[0]

    def next(self, last_tokens):
        # allow_tokens 为None表示不限制
        over = False
        (
            block_over,
            self.belonging_task.allow_tokens,
            self.belonging_task.direct_infer_tokens,
        ) = self.now_block.next(last_tokens)
        if block_over:
            idx = self.constraint_blocks.index(self.now_block)
            if idx + 1 < len(self.constraint_blocks):
                self.now_block = self.constraint_blocks[idx + 1]
                self.belonging_task.allow_tokens = self.now_block.init_allow_tokens
            else:
                over = True
        return over


class ConstraintGenerateBlock:
    def __init__(self, rule, **kwargs):
        self.init_allow_tokens = None
        self.init_direct_infer_tokens = None
        self.rule = rule
        if rule == "generate":
            self.max_tokens = kwargs["max_tokens"]
            self.token_stop = kwargs["token_stop"]
            self.n_now = 0
        elif rule == "select":
            self.selections = kwargs["selections"]
            self.selections = [
                global_config.tokenizer_eval.encode(x)
                for x in self.selections
                if x.strip()
            ]
            self.init_allow_tokens = [s[0] for s in self.now_block.selections]
        elif rule == "str":
            self.text = kwargs["text"]
            self.text_tokens = global_config.tokenizer_eval.encode(self.text)
            self.init_direct_infer_tokens = self.text_tokens

    def next(self, last_tokens):
        over = False
        if self.rule == "generate":
            self.n_now += len(last_tokens)
            if self.n_now >= self.max_tokens or last_tokens[-1] in self.token_stop:
                over = True
            return over, None, None
        elif self.rule == "select":
            self.selections = [
                s[1:]
                for s in self.selections
                if s and last_tokens[-1] == s[0] and s[1:]
            ]
            allow_tokens = [s[0] for s in self.selections]
            return len(self.selections) == 0, allow_tokens, None
        elif self.rule == "str":
            return True, None, self.text_tokens


class CollateInferenceTokens:
    def __init__(self):
        self.token_state_dict = OrderedDict()

    def update(self, idx, token: int, state):
        self.token_state_dict[idx] = (token, state)

    @property
    def batch(self):
        idx_list = list(self.token_state_dict.keys())
        token_state_list = list(self.token_state_dict.values())  # [(token, states)]
        states = token_state_list[0][1]
        tokens = [[token_state_list[0][0]]]
        for token, state in token_state_list[1:]:
            # 需要处理state为None的情况
            states = states + state
            tokens.append([token])
        return idx_list, tokens, states


class BatchingTask:
    def __init__(
        self,
        user_id,
        state,
        begin_with_tokens,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
        penalty_decay,
        constraint_str,
        token_stop,
        token_stop_supp,
        max_tokens,
        token_ban=[],
        occurence={},
    ):
        self.user_id = user_id
        self.begin_with_tokens = begin_with_tokens
        self.temp = temp
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.penalty_decay = penalty_decay
        self.constraint_str = constraint_str
        self.token_stop = token_stop
        self.token_stop_supp = token_stop_supp
        self.max_tokens = max_tokens
        self.token_ban = token_ban

        self.state = state
        self.in_ppls = []
        self.out_ppls = []
        self.resp_tokens = []
        self.iter_tokens = []
        self.last_out = None

        self.rule_blocks = self.init_rule_blocks()
        self.allow_tokens = self.rule_blocks.now_block.init_allow_tokens
        self.direct_infer_tokens = self.rule_blocks.now_block.init_direct_infer_tokens

        self.occurence = occurence

    def reach_max_tokens(self):
        return len(self.resp_tokens) >= self.max_tokens

    def init_rule_blocks(self):
        return ConstraintGenerateBlockList(self, self.constraint_str)


class BatchingInferenceHelper:
    def __init__(
        self,
        batch_infer_func,
        max_bsz=5,
        broadcast_host: str = "0.0.0.0",
        broadcast_port: int = 4516,
    ):
        self.wait_for_inference_tasks = []  # state_id, tokens_list
        self.batch_infer_func = batch_infer_func
        self.max_bsz = max_bsz
        self.tokenizer = global_config.tokenizer_eval
        self.batch_tasks = []
        # 创建 UDP 客户端 socket
        self.broadcast_host = broadcast_host
        self.broadcast_port = broadcast_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_BROADCAST, 1
        )  # 启用广播
        self.broadcast_address = (
            broadcast_host,
            broadcast_port,
        )  # 使用广播地址（例如：255.255.255.255）

    def add_task(
        self,
        user_id,
        state,
        begin_with_tokens,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
        penalty_decay,
        constraint_str,
        token_stop,
        token_stop_supp,
        max_tokens=500,
        token_ban=[],
        occurence={},
    ):
        task = BatchingTask(
            user_id,
            state,
            begin_with_tokens,
            temp,
            top_p,
            presence_penalty,
            frequency_penalty,
            penalty_decay,
            constraint_str,
            token_stop,
            token_stop_supp,
            max_tokens,
            token_ban,
            occurence,
        )
        self.wait_for_inference_tasks.append(task)
        return task

    def init_task(self, task: BatchingTask):
        tokens, states = task.begin_with_tokens, task.state
        out, new_states = self.batch_infer_func([tokens], states)
        task.in_ppls = ppl(tokens[1:], out[0, :-1, :])
        task.state = new_states
        task.last_out = out
        # # 这里要拿到第一批的allow_tokens
        # task.iter_tokens= self.sample_logits(task,xxxxx,task.token_ban)

    @torch.no_grad()
    def step(self):
        if self.batch_tasks:
            over_and_success_list = []
            batch_tokens = CollateInferenceTokens()
            for i, task in enumerate(self.batch_tasks):
                # 采样
                # try:
                if task.direct_infer_tokens:
                    out, new_states = self.batch_infer_func(
                        [task.direct_infer_tokens], task.state
                    )
                    task.resp_tokens += task.direct_infer_tokens
                    task.iter_tokens = task.direct_infer_tokens
                    task.out_ppls += ppl(task.direct_infer_tokens[1:], out[0, :-1, :])
                    task.state = new_states
                    task.last_out = out
                else:
                    token, token_ppl = self.sample_logits(
                        task,
                        task.allow_tokens,
                        task.token_ban,
                    )
                    task.resp_tokens.append(token)
                    task.out_ppls += token_ppl
                    task.iter_tokens = [token]
                    batch_tokens.update(i, token, task.state)
                over = task.rule_blocks.next(task.iter_tokens)

                batch_out, new_states = self.batch_infer_func(
                    batch_tokens.batch[1], batch_tokens.batch[2]
                )
                batch_indices = batch_tokens.batch[0]
                new_states_list = new_states.unbind()
                out_list = torch.unbind(batch_out, dim=0)
                out_list = [x.unsqueeze(0) for x in out_list]
                over_and_success_list.append((over, True))
            # except Exception as e:
            #     print(
            #         f"编号{task.user_id}的任务出现异常，已被移除。\n错误信息：{e}"
            #     )
            #     over_and_success_list.append((True, False))
            #     batch_indices = [i]
            #     new_states_list = [task.state]
            #     out_list = [task.last_out]

            returns = {}
            for i, task in enumerate(self.batch_tasks):
                if i in batch_indices:
                    task.state = new_states_list[batch_indices[i]]
                    task.last_out = out_list[batch_indices[i]]
                    returns[task.user_id] = {
                        # "resp_tokens": task.resp_tokens,
                        # "iter_tokens": task.iter_tokens,
                        # "in_ppls": task.in_ppls,
                        # "out_ppls": task.out_ppls,
                        # "occurence": task.occurence,
                        "over": over_and_success_list[i][0],
                        "success": over_and_success_list[i][1],
                        "new": True,
                    }
                # if over_and_success_list[i][0]:
                #     self.batch_tasks.remove(task)
            messages = json.dumps(returns, ensure_ascii=False)
            self.client_socket.sendto(messages.encode(), self.broadcast_address)

            return returns

    def sample_logits(
        self,
        task: BatchingTask,
        allow_tokens,
        ban_tokens,
    ):
        out = task.last_out
        occurrence = task.occurence
        for n in occurrence:
            out[0, -1, n] -= (
                task.presence_penalty + occurrence[n] * task.frequency_penalty
            )
        if allow_tokens:
            mask = torch.full_like(out[0, -1, :], -1e38, device=out.device)
            mask[allow_tokens] = out[0, -1, allow_tokens]
            out[0, -1, :] = mask
        for ban_token in ban_tokens:
            out[0, -1, ban_token] = torch.full_like(
                out[0, -1, ban_token], -1e38, device=out.device
            )
        token = sample_logits(
            out[0, -1, :],
            temperature=task.temp,
            top_p=task.top_p,
        )
        token_ppl = ppl([token], out[0, -1:, :])
        for xxx in occurrence:
            occurrence[xxx] *= task.penalty_decay
        if 49 <= token <= 58:  # numbers
            pass
        elif token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        task.occurence = occurrence
        return token, token_ppl

    def loop_step(self):
        if len(self.batch_tasks) < self.max_bsz and self.wait_for_inference_tasks:
            new_task = self.wait_for_inference_tasks.pop(0)
            self.batch_tasks.append(new_task)
            self.init_task(new_task)
        if self.batch_tasks:
            self.step()
