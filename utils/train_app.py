import os
from config import global_config

train_config = global_config.train_service_config

import gc
import math
import copy
import torch
import deepspeed
from utils.message_manager import cList, Conversation
import requests
import json
from config import RWKV
from config import BlockStateList
import torch.nn.functional as F
from RWKV.functions import (
    train_forward,
    train_forward_from_embds,
    speak,
    ppl,
    speak_next_token,
    calc_cross_entropy_loss,
)
from RWKV.multimodal_functions import (
    voice_encode_and_adapt,
)
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import random
import wandb
import sys
import psutil

from utils.collections import pad_2d_list_with_zeros, pad_and_batch
from utils.dataset.dataset import MultimodalDataset, read_bin_wav
from utils.dataset.dataset_functions import (
    UnitStreamProcessor,
    TraversalDataloader,
    MyDataloader,
    EpochSampleDataloader,
)

from config import vocoder

from typing import List
from functools import partial
import argparse

from deepspeed import comm as dist

deepspeed.init_distributed()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload_model_dir", type=str, help="重新加载训练模型位置")
    parser.add_argument("--lr_init", type=float, help="初始学习率")
    parser.add_argument("--lr_final", type=float, help="最终学习率")
    parser.add_argument("--warmup_steps", type=int, help="warmup步数")
    args = parser.parse_known_args()[0]
    return args


class OnlineTrainingAPP:
    def __init__(self):
        self.args = train_config
        cmd_args = get_args()
        if cmd_args.reload_model_dir:
            print(f"overwrite model dir: {cmd_args.reload_model_dir}")
            train_config.model.load_model = cmd_args.reload_model_dir
        if cmd_args.lr_init:
            print(f"overwrite lr_init: {cmd_args.lr_init}")
            self.args.train.lr_init = float(cmd_args.lr_init)
        if cmd_args.lr_final:
            print(f"overwrite lr_final: {cmd_args.lr_final}")
            self.args.train.lr_final = float(cmd_args.lr_final)
        if cmd_args.warmup_steps:
            print(f"overwrite warmup_steps: {cmd_args.warmup_steps}")
            self.args.train.warmup_steps = int(cmd_args.warmup_steps)
        print(f"load model from: {train_config.model.load_model}")
        self.model = RWKV(self.args, global_config.voice_on)
        self.model_engine = self.build_engine(
            lr_init=self.args.train.lr_init,
            lr_final=self.args.train.lr_final,
            warmup_steps=self.args.train.warmup_steps,
        )
        self.infer_tokenizer = global_config.tokenizer_eval
        self.train_tokenizer = global_config.tokenizer_train

        self.train_state = None  # 训练时的滚动state
        self.online_lr_state = None  # 在线学习state

        self.rank = dist.get_rank()
        self.total_gpus = dist.get_world_size()

        if global_config.wandb_proj and self.rank == 0:
            wandb.init(project=global_config.wandb_proj)

        if global_config.voice_on:
            self.feat_extractor = torch.load(
                train_config.vocoder.load_feature_extractor_dir
            )
            self.voice_losses = vocoder.Losses(train_config)
            self.discriminator = vocoder.VocoderDiscriminator(train_config.vocoder)
            self.discriminator_engine, _ = self.discriminator.build_engine()
            self.voice_unit_len = (
                self.args.vocoder.head.hop_length * self.args.vocoder.adapter.chunk_len
            )

    def load_model(
        self,
        ckpt_dir: str,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        print(f"从{ckpt_dir}重新读取模型...")
        if hasattr(self, "model_engine"):
            current_process = psutil.Process()
            parent_process = current_process.parent().parent()

            cmd_line = parent_process.cmdline()
            ds_idx, ds = next(
                (
                    (idx, item)
                    for idx, item in enumerate(cmd_line)
                    if item.endswith("/deepspeed")
                ),
                None,
            )
            cmd = ["deepspeed"] + parent_process.cmdline()[ds_idx + 1 :]
            cmd += [
                "--reload_model_dir",
                ckpt_dir,
                "--lr_init",
                str(lr_init) if lr_init else str(self.args.train.lr_init),
                "--lr_final",
                str(lr_final) if lr_final else str(self.args.train.lr_final),
                "--warmup_steps",
                (
                    str(warmup_steps)
                    if warmup_steps
                    else str(self.args.train.warmup_steps)
                ),
            ]
            print("====================================================")
            print("run", " ".join(cmd))
            print("====================================================")
            os.execv(ds, cmd)

    def save_weight(
        self, name: str, save_train_state: bool = False, folder: str = None
    ):
        folder = folder if folder else global_config.ckpt_dir
        fpath = f"{folder}/{name}.pth"
        state_path = f"{folder}/{name}.state"
        self.model.load_state_dict(self.model_engine.module.state_dict())
        torch.save(self.model.state_dict(), fpath)
        if save_train_state and self.train_state is not None:
            torch.save(self.train_state, state_path)
        gc.collect()
        torch.cuda.empty_cache()
        return fpath

    def save_vocoder(self, name: str):
        fpath_gen = f"{global_config.ckpt_dir}/{name}_vocoder_gen.pth"
        temp_state_dict = {
            k.replace("module.", ""): v
            for k, v in self.vocoder_engine.gen_engine.state_dict().items()
        }
        self.vocoder_engine.model.generator.load_state_dict(temp_state_dict)
        torch.save(self.vocoder_engine.model.generator.state_dict(), fpath_gen)
        fpath_disc = f"{global_config.ckpt_dir}/{name}_vocoder_disc.pth"
        if train_config.vocoder.GAN_on:
            temp_state_dict = {
                k.replace("module.", ""): v
                for k, v in self.vocoder_engine.disc_engine.state_dict().items()
            }
            self.vocoder_engine.model.discriminators.load_state_dict(temp_state_dict)
            torch.save(
                self.vocoder_engine.model.discriminators.state_dict(), fpath_disc
            )
        gc.collect()
        torch.cuda.empty_cache()

    def train_text_from_messages(
        self,
        messages: List[cList],
        batch_size: int = 1,
        n_save_ckpt: int = -1,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = None,
        keep_train_states: bool = False,
        use_ego_mask: bool = False,
        ignore_ctx: bool = False,
    ):
        assert batch_size % self.total_gpus == 0
        dp_chunk_len = batch_size // self.total_gpus
        """
        单batch文本训练
        """
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )
        multi_scale_ctx = (
            self.args.model.ctx_len if multi_scale_ctx is None else multi_scale_ctx
        )
        multi_scale_alpha = (
            self.args.train.multi_scale_alpha
            if multi_scale_alpha is None
            else multi_scale_alpha
        )
        assert multi_scale_ctx * multi_scale_alpha > 2
        assert 0 < multi_scale_alpha <= 1
        self.model_engine.train()
        all_tokens = []
        all_masks = []
        losses = []
        for clist in messages:
            tokens, masks = clist.to_tokens(
                self.train_tokenizer.encode, use_ego_mask=use_ego_mask
            )
            all_tokens += tokens
            all_masks += masks
            all_tokens += [0]
            all_masks += [0]
        all_tokens = pad_and_batch(all_tokens, batch_size)
        all_masks = pad_and_batch(all_masks, batch_size)
        dp_tokens = all_tokens[
            self.rank * dp_chunk_len : (self.rank + 1) * dp_chunk_len
        ]
        dp_masks = all_masks[self.rank * dp_chunk_len : (self.rank + 1) * dp_chunk_len]
        assert len(dp_masks) == len(dp_tokens)
        for step, (mean_loss, train_tokens) in enumerate(
            self.learn_tokens(
                tokens=dp_tokens,
                masks=dp_masks,
                min_loss=min_loss,
                min_loss_fix=min_loss_fix,
                max_loss=max_loss,
                max_loss_fix=max_loss_fix,
                states=None,
                multi_scale_ctx=multi_scale_ctx,
                multi_scale_alpha=multi_scale_alpha,
                keep_train_states=keep_train_states,
                ignore_ctx=ignore_ctx,
            ),
            1,
        ):
            # for b in range(batch_size):
            #     while 0 in train_tokens[b]:
            #         train_tokens[b].remove(0)
            print(f"gpu{self.rank}: mean-loss->{mean_loss}")
            print(
                f"gpu{self.rank}->{self.infer_tokenizer.decode(train_tokens)[self.rank][:45]}"
            )
            gc.collect()
            torch.cuda.empty_cache()
            if self.rank == 0:
                if global_config.wandb_proj:
                    wandb.log({"text_loss": mean_loss})
                if n_save_ckpt > 0 and step % n_save_ckpt == 0:
                    print(f"====save at step={step}====")
                    self.save_weight(f"train_folder_step={step}")
            losses.append(mean_loss)
        mean_loss = sum(losses) / len(losses)
        return mean_loss

    def learn_tokens(
        self,
        tokens: list,
        masks: list,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        states: BlockStateList = None,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = 1,
        keep_train_states: bool = False,
        ignore_ctx: bool = False,
        return_left_token: bool = False,
    ):
        """
        文本训练
        """
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )
        multi_scale_ctx = (
            self.args.model.ctx_len if multi_scale_ctx is None else multi_scale_ctx
        )
        multi_scale_alpha = (
            self.args.train.multi_scale_alpha
            if multi_scale_alpha is None
            else multi_scale_alpha
        )
        self.model_engine.train()
        assert multi_scale_ctx * multi_scale_alpha > 2
        assert 0 < multi_scale_alpha <= 1
        assert len(tokens) != 0
        total = 0
        mean_loss = 0
        i = 0

        tokens = torch.tensor(tokens, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.float32)
        assert tokens.shape[1] != 0
        while tokens.shape[1] > 0:
            i += 1
            ctx_len = (
                random.randint(
                    int(multi_scale_ctx * multi_scale_alpha),
                    multi_scale_ctx,
                )
                if not ignore_ctx
                else 99999999999999999
            )

            output = tokens[:, :ctx_len]
            output_masks = masks[:, :ctx_len]
            tokens = tokens[:, ctx_len - 1 :]
            masks = masks[:, ctx_len - 1 :]
            if not keep_train_states:
                states = None
            batch_tokens = copy.deepcopy(output).to(
                next(self.model_engine.parameters()).device
            )
            batch_masks = copy.deepcopy(output_masks).to(
                next(self.model_engine.parameters()).device
            )
            print(batch_tokens.shape, batch_masks.shape)
            m, states = train_forward(
                self.model_engine, batch_tokens, batch_masks, states
            )
            self.train_state = states
            loss = m.item()
            print(f"loss={m}")
            if loss < min_loss:
                m = m * min_loss_fix
                print(f"(<min)fixed_loss:{m}")
            elif loss > max_loss:
                print(f"(>max)before_fixed_loss:{m}")
                m = m * max_loss_fix
                print(f"(>max)fixed_loss:{m}")
            self.model_engine.backward(m)
            self.model_engine.step()
            total += loss
            mean_loss = total / i
            if tokens.shape[1] == 0:
                break
            if return_left_token:
                yield mean_loss, output, tokens.shape[1]
            else:
                yield mean_loss, output

    def train_from_folder(
        self,
        forder_dir: str,
        epoch: int,
        batch_size_per_gpu: int = 1,
        n_save_ckpt: int = 1,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = None,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        n_save_step: int = None,
        keep_states_mode: str = "never",
        dataloader_workers_per_gpu: int = 2,
        begin_with_state_dir=None,
        use_qa_mask: bool = False,
    ):
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )

        multi_scale_ctx = (
            self.args.model.ctx_len if multi_scale_ctx is None else multi_scale_ctx
        )
        multi_scale_alpha = (
            self.args.train.multi_scale_alpha
            if multi_scale_alpha is None
            else multi_scale_alpha
        )
        assert multi_scale_ctx * multi_scale_alpha > 2
        assert 0 < multi_scale_alpha <= 1
        self.model_engine.train()

        assert keep_states_mode in ["never", "step", "epoch"]

        self.train_state = (
            None if begin_with_state_dir is None else torch.load(begin_with_state_dir)
        )
        # check batch state
        if self.train_state is not None:
            n_state_batch = self.train_state.shift_states.shape[2]
            total_batch = batch_size_per_gpu * self.total_gpus
            if total_batch > n_state_batch:
                alpha = (n_state_batch + total_batch - 1) // n_state_batch
                self.train_state = self.train_state.duplicate(alpha)[:n_state_batch]
            elif total_batch < n_state_batch:
                print(
                    "警告: 训练的batch数量小于读取state的batch数量, 可能会导致信息损失。"
                )
                self.train_state = self.train_state[:total_batch]

        total_text_loss = []

        dataset = MultimodalDataset(
            dataset_dir=forder_dir,
            tokenizer=self.train_tokenizer,
            voice_read_func=None if not global_config.voice_on else read_bin_wav,
            video_load_func=None,
            ctx_len=multi_scale_ctx,
            qa_mask_on=use_qa_mask,
        )

        dataloader = TraversalDataloader(
            dataset=dataset,
            batch_size=batch_size_per_gpu,
            num_workers=dataloader_workers_per_gpu,
            multi_scale_alpha=multi_scale_alpha,
        )

        stream_processor = UnitStreamProcessor(train_config)

        for e in range(epoch):
            print(f"====gpu:{self.rank},train epoch:{e}====")
            if keep_states_mode == "step":
                self.train_state = None
            for step, (batch_units, batch_masks) in enumerate(dataloader):
                unit_dicts = stream_processor.encode(
                    self.model_engine,
                    self.feat_extractor if global_config.voice_on else None,
                    batch_units,
                    voice_encode_and_adapt,
                    device=next(self.model_engine.parameters()).device,
                    dtype=next(self.model_engine.parameters()).dtype,
                )
                dist.barrier()
                embds = unit_dicts["main"]
                token_targets = torch.tensor(
                    unit_dicts["tokens_target"],
                    dtype=torch.long,
                    device=next(self.model_engine.parameters()).device,
                )
                token_masks = (token_targets != 0).long()
                batch_masks = torch.tensor(
                    batch_masks,
                    device=next(self.model_engine.parameters()).device,
                    dtype=torch.float32,
                )[:, 1:]
                mask_targets = token_masks * batch_masks
                if keep_states_mode == "never":
                    self.train_state = None
                out_latent, out_logits, self.train_state = (
                    self.model_engine.forward_from_embeddings(
                        embds[:, :-1, :], self.train_state
                    )
                )
                text_loss = calc_cross_entropy_loss(
                    out_logits, token_targets, mask_targets
                )
                text_loss_item = text_loss.item()
                if text_loss < min_loss:
                    text_loss = text_loss * min_loss_fix
                    print(f"(<min)fixed_loss:{text_loss}")
                elif text_loss > max_loss:
                    print(f"(>max)before_fixed_loss:{text_loss}")
                    text_loss = text_loss * max_loss_fix
                    print(f"(>max)fixed_loss:{text_loss}")
                # TODO: 音频loss
                #
                # ===============
                m = text_loss
                self.model_engine.zero_grad()
                self.model_engine.backward(m)
                self.model_engine.step()
                total_text_loss.append(text_loss_item)
                total_text_loss = total_text_loss[-100:]
                mean_text_loss = sum(total_text_loss) / len(total_text_loss)

                yield json.dumps(
                    {
                        "epoch": e,
                        "step": step,
                        "mean_text_loss": mean_text_loss,
                        "text_loss": text_loss_item,
                        "n_tokens": dataloader.n_dataset_ctx,
                        "left_tokens": dataloader.n_dataset_ctx
                        - dataloader.current_ctx,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(
                    f"gpu{self.rank}: mean-text-loss->{mean_text_loss} | now-text-loss->{text_loss_item}"
                )

                dist.barrier()
                gc.collect()
                torch.cuda.empty_cache()
                if self.rank == 0:
                    if global_config.wandb_proj:
                        wandb.log({"mean_text_loss": mean_text_loss})

                    if n_save_step and step % n_save_step == 0:
                        print(f"====epoch: {e},save at step={step}====")
                        self.save_weight(f"train_single_folder_epoch={e}_step={step}")

            if e % n_save_ckpt == 0 and self.rank == 0:
                svpath = self.save_weight(
                    f"train_single_folder_epoch={e}", save_train_state=True
                )
                yield json.dumps(
                    {
                        "over": False,
                        "to_dir": svpath,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(f"====save at epoch={e}====")
        yield json.dumps(
            {
                "over": True,
                "to_dir": svpath,
            },
            ensure_ascii=False,
        ) + "\n"

    def train_from_folders(
        self,
        folder_weight_dir_list,
        epoch: int,
        batch_size_per_gpu: int = 1,
        n_save_ckpt: int = 1,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        n_save_step: int = None,
        dataloader_workers_per_gpu: int = 2,
        use_qa_mask: bool = False,
    ):
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )

        self.model_engine.train()

        total_text_loss = []

        dataset_folder_list = [
            folder_dir for folder_dir, n_sample_lines in folder_weight_dir_list
        ]
        n_sample_list = [
            n_sample_lines for folder_dir, n_sample_lines in folder_weight_dir_list
        ]

        epoch_sample_dataloader = EpochSampleDataloader(
            dataset_folder_list,
            n_sample_list,
            batch_size_per_gpu,
            num_workers=dataloader_workers_per_gpu,
            tokenizer=self.train_tokenizer,
            voice_read_func=None if not global_config.voice_on else read_bin_wav,
            video_load_func=None,
            ctx_len=self.args.model.ctx_len,
            total_epoch=epoch,
            use_qa_mask=use_qa_mask,
        )
        stream_processor = UnitStreamProcessor(train_config)

        for e, (epoch_units, epoch_masks) in enumerate(epoch_sample_dataloader):
            print(f"====gpu:{self.rank},train epoch:{e}====")
            n_data = len(epoch_units)
            for step, (batch_units, batch_masks) in enumerate(
                zip(epoch_units, epoch_masks)
            ):
                unit_dicts = stream_processor.encode(
                    self.model_engine,
                    self.feat_extractor if global_config.voice_on else None,
                    batch_units,
                    voice_encode_and_adapt,
                    device=next(self.model_engine.parameters()).device,
                    dtype=next(self.model_engine.parameters()).dtype,
                )
                dist.barrier()
                embds = unit_dicts["main"]
                token_targets = torch.tensor(
                    unit_dicts["tokens_target"],
                    dtype=torch.long,
                    device=next(self.model_engine.parameters()).device,
                )
                token_masks = (token_targets != 0).long()
                batch_masks = torch.tensor(
                    batch_masks,
                    device=next(self.model_engine.parameters()).device,
                    dtype=torch.float32,
                )[:, 1:]
                mask_targets = token_masks * batch_masks
                out_latent, out_logits, self.train_state = (
                    self.model_engine.forward_from_embeddings(
                        embds[:, :-1, :], self.train_state
                    )
                )
                text_loss = calc_cross_entropy_loss(
                    out_logits, token_targets, mask_targets
                )
                text_loss_item = text_loss.item()
                if text_loss < min_loss:
                    text_loss = text_loss * min_loss_fix
                    print(f"(<min)fixed_loss:{text_loss}")
                elif text_loss > max_loss:
                    print(f"(>max)before_fixed_loss:{text_loss}")
                    text_loss = text_loss * max_loss_fix
                    print(f"(>max)fixed_loss:{text_loss}")
                # TODO: 音频loss
                #
                # ===============
                m = text_loss
                self.model_engine.zero_grad()
                self.model_engine.backward(m)
                self.model_engine.step()
                total_text_loss.append(text_loss_item)
                total_text_loss = total_text_loss[-100:]
                mean_text_loss = sum(total_text_loss) / len(total_text_loss)

                yield json.dumps(
                    {
                        "epoch": e,
                        "step": step,
                        "mean_text_loss": mean_text_loss,
                        "text_loss": text_loss_item,
                        "n_data": n_data,
                        "left_data": n_data - step - 1,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(
                    f"gpu{self.rank}: mean-text-loss->{mean_text_loss} | now-text-loss->{text_loss_item}"
                )

                dist.barrier()
                gc.collect()
                torch.cuda.empty_cache()
                if self.rank == 0:
                    if global_config.wandb_proj:
                        wandb.log({"mean_text_loss": mean_text_loss})

                    if n_save_step and step % n_save_step == 0:
                        print(f"====epoch: {e},save at step={step}====")
                        self.save_weight(f"train_single_folder_epoch={e}_step={step}")

                if step >= n_data - 1:
                    break
            if e % n_save_ckpt == 0 and self.rank == 0:
                svpath = self.save_weight(
                    f"train_single_folder_epoch={e}", save_train_state=True
                )
                yield json.dumps(
                    {
                        "over": False,
                        "to_dir": svpath,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(f"====save at epoch={e}====")
        yield json.dumps(
            {
                "over": True,
                "to_dir": svpath,
            },
            ensure_ascii=False,
        ) + "\n"

    def _get_batch_logps(
        self,
        logits,
        labels: torch.LongTensor,
        input_mask=None,
        average_log_prob: bool = True,
    ):
        logits = logits.to(
            device=next(self.model_engine.parameters()).device,
            dtype=self.args.model.dtype,
        )
        labels = labels.to(
            device=next(self.model_engine.parameters()).device,
        )
        if input_mask is not None:
            input_mask = input_mask.to(
                device=next(self.model_engine.parameters()).device,
            )

        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != 0

        # dummy token; we'll ignore the losses on these tokens later
        if input_mask is not None:
            mask = input_mask[:, 1:]
            loss_mask[mask == 0] = False
            labels[mask == 0] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def build_engine(self, lr_init, lr_final, warmup_steps):
        if hasattr(self, "model_engine"):
            print("重新构建模型引擎...")
            for param_group in self.model_engine.optimizer.param_groups:
                param_group["lr"] = lr_init
            self.lr_scheduler.warmup_min_lr = lr_init  # 更新学习率调度器的初始学习率
            self.lr_scheduler.warmup_max_lr = lr_final  # 更新学习率调度器的最终学习率
            self.lr_scheduler.warmup_num_steps = warmup_steps  # 更新学习率调度器的预热步数
            if hasattr(self.lr_scheduler, 'last_epoch'):
                self.lr_scheduler.last_epoch = -1  # 重置调度器状态
            if hasattr(self.lr_scheduler,"min_lrs"):
                for i in range(len(self.lr_scheduler.min_lrs)):
                    self.lr_scheduler.min_lrs[i]=lr_init
            if hasattr(self.lr_scheduler,"max_lrs"):
                for i in range(len(self.lr_scheduler.max_lrs)):
                    self.lr_scheduler.max_lrs[i]=lr_final
            self.lr_scheduler.step()
            print("=========================================")
            for attr, value in vars(self.lr_scheduler).items():
                print(f"{attr}: {value}")
            print(self.lr_scheduler.warmup_min_lr)
            return self.model_engine
        else:
            ds_config = {
                "bfloat16": {"enabled": "auto"},
                "gradient_accumulation_steps": self.args.deepspeed.gradient_accumulation_steps,
                "gradient_clipping": self.args.train.grad_cp,
                "train_micro_batch_size_per_gpu": 1,
            }
            if train_config.deepspeed.zero:
                ds_config["zero_optimization"] = {
                    "stage": train_config.deepspeed.ds_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e6,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e6,
                    "contiguous_gradients": True,
                }

                if train_config.deepspeed.offload_optimizer:
                    ds_config["zero_optimization"]["offload_optimizer"] = {
                        "device": "cpu",
                        "pin_memory": True,
                    }
                if (
                    train_config.deepspeed.offload_param_stage3
                    and train_config.deepspeed.ds_stage == 3
                ):
                    ds_config["zero_optimization"]["offload_param"] = {
                        "device": "cpu",
                        "pin_memory": True,
                    }

            self.optimizer = (
                DeepSpeedCPUAdam(
                    self.model.get_optim_groups(),
                    lr=lr_init,
                    betas=(self.args.train.beta1, self.args.train.beta2),
                    eps=self.args.train.adam_eps,
                    adamw_mode=self.args.train.adamw_mode,
                    weight_decay=self.args.train.weight_decay,
                    amsgrad=False,
                    bias_correction=True,
                )
                if train_config.deepspeed.zero and train_config.deepspeed.offload_optimizer
                else FusedAdam(
                    self.model.get_optim_groups(),
                    lr=lr_init,
                    betas=(self.args.train.beta1, self.args.train.beta2),
                    eps=self.args.train.adam_eps,
                    bias_correction=True,
                    adam_w_mode=self.args.train.adamw_mode,
                    weight_decay=self.args.train.weight_decay,
                    amsgrad=False,
                )
            )

            self.lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                self.optimizer,
                warmup_min_lr=lr_init,
                warmup_max_lr=lr_final,
                warmup_num_steps=warmup_steps,
                warmup_type="linear",
            )

            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=ds_config,
            )
            print("cuda available", torch.cuda.device_count())
        return self.model_engine
