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
    calc_cross_entropy_loss,
)
from RWKV.multimodal_functions import (
    voice_encode_and_adapt,
)
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import random
import wandb


from utils.collections import pad_2d_list_with_zeros,pad_and_batch
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

from deepspeed import comm as dist

deepspeed.init_distributed()


class OnlineTrainingAPP:
    def __init__(self):
        self.args = train_config
        print(f"load model from: {train_config.model.load_model}")
        self.model = RWKV(self.args, global_config.voice_on)
        self.model_engine = self.build_engine()
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

    def load_model(self, ckpt_dir: str):
        model_weights = torch.load(ckpt_dir, map_location="cpu")
        self.model.load_state_dict(model_weights, strict=False)

        del self.model_engine
        gc.collect()
        torch.cuda.empty_cache()

        self.model_engine = self.build_engine()

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
        min_loss: float=None,
        max_loss: float=None,
        min_loss_fix: float=None,
        max_loss_fix: float=None,
        states: BlockStateList = None,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = 1,
        keep_train_states:bool=False,
        ignore_ctx:bool=False,
        return_left_token:bool=False,
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
        epoch:int,
        batch_size_per_gpu:int=1,
        n_save_ckpt:int=1,
        min_loss:float=None,
        max_loss:float=None,
        min_loss_fix:float=None,
        max_loss_fix:float=None,
        n_save_step:int=None,
        dataloader_workers_per_gpu:int=2,
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

    def calc_dpo_losses_multilabel(
        self,
        logits_list,
        logits_ref_list,
        score_list,
        batch_tokens_list,
        batch_masks_list,
        threshold_score:float=2.5,
        beta:float=0.5,
    ):
        def sign(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0

        assert (
            len(logits_list)
            == len(logits_ref_list)
            == len(score_list)
            == len(batch_tokens_list)
            == len(batch_masks_list)
        )
        total_pos_abs_weight = 0
        total_neg_abs_weight = 0
        pos_upper = None
        neg_upper = None
        min_score, max_score, best_lp, worse_lp, best_ref_lp, worse_ref_lp = (
            114514,
            -114514,
            None,
            None,
            None,
            None,
        )
        for logits, logits_ref, score, tokens, masks in zip(
            logits_list,
            logits_ref_list,
            score_list,
            batch_tokens_list,
            batch_masks_list,
        ):
            is_postive = score > threshold_score
            lp_ref = self._get_batch_logps(
                logits_ref,
                torch.tensor([tokens], dtype=torch.long),
                torch.tensor([masks], dtype=torch.long),
            )
            lp = self._get_batch_logps(
                logits,
                torch.tensor([tokens], dtype=torch.long),
                torch.tensor([masks], dtype=torch.long),
            )
            if score > max_score:
                max_score = score
                best_lp = lp.detach()
                best_ref_lp = lp_ref
            if score < min_score:
                min_score = score
                worse_lp = lp.detach()
                worse_ref_lp = lp_ref
            abs_weight = math.sqrt(abs(score - threshold_score))
            if is_postive:
                total_pos_abs_weight += abs_weight
                if pos_upper is None:
                    pos_upper = abs_weight * (lp - lp_ref)
                else:
                    pos_upper += abs_weight * (lp - lp_ref)
            else:
                total_neg_abs_weight += abs_weight
                if neg_upper is None:
                    neg_upper = abs_weight * (lp - lp_ref)
                else:
                    neg_upper += abs_weight * (lp - lp_ref)
        pos = pos_upper / total_pos_abs_weight
        neg = neg_upper / total_neg_abs_weight

        losses = -F.logsigmoid(beta * (pos - neg))
        chosen_rewards = beta * (best_lp - best_ref_lp).detach()
        rejected_rewards = beta * (worse_lp - worse_ref_lp).detach()
        return losses, chosen_rewards, rejected_rewards

    def calc_dpo_losses(
        self,
        logits_p,
        logits_ref_p,
        logits_n,
        logits_ref_n,
        batch_pos_tokens,
        batch_neg_tokens,
        batch_pos_masks,
        batch_neg_masks,
        beta=0.5,
    ):
        lp_ref_p = self._get_batch_logps(
            logits_ref_p,
            torch.tensor(batch_pos_tokens, dtype=torch.long),
            torch.tensor(batch_pos_masks, dtype=torch.long),
        )
        lp_ref_n = self._get_batch_logps(
            logits_ref_n,
            torch.tensor(batch_neg_tokens, dtype=torch.long),
            torch.tensor(batch_neg_masks, dtype=torch.long),
        )
        lp_p = self._get_batch_logps(
            logits_p,
            torch.tensor(batch_pos_tokens, dtype=torch.long),
            torch.tensor(batch_pos_masks, dtype=torch.long),
        )
        lp_n = self._get_batch_logps(
            logits_n,
            torch.tensor(batch_neg_tokens, dtype=torch.long),
            torch.tensor(batch_neg_masks, dtype=torch.long),
        )

        losses = -F.log_softmax(beta * (lp_p - lp_ref_p - lp_n + lp_ref_n))
        chosen_rewards = beta * (lp_p - lp_ref_p).detach()
        rejected_rewards = beta * (lp_n - lp_ref_n).detach()

        return losses, chosen_rewards, rejected_rewards

    def train_dpo_v2(
        self,
        folder_weight_dir_list,
        inference_service_server="http://localhost:4514/",
        step_save_ckpt=None,
        allow_multilabel=True,
        n_use_max_choices=5,
    ):
        dataset = FolderDPODatasetV2(folder_weight_dir_list)

        def test_server(server):
            try:
                response = requests.get(server + "/test")
                if response.status_code == 200:
                    return True
                else:
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Exception occurred: {e}")
                return False

        assert test_server(inference_service_server)

        resp = requests.post(
            f"{inference_service_server}/regist_state_id",
            json={},
        ).json()
        state_idx = resp["access_token"]

        for step, ordered_data_dicts in enumerate(dataset.load_batch_datas(), 0):
            # 每个新对话，重置state
            state = None
            resp = requests.post(
                f"{inference_service_server}/reset_state_id",
                json={"access_token": state_idx},
            ).json()
            for turn_choice_dict in ordered_data_dicts.values():
                if len(turn_choice_dict) > 1:
                    best_clist = None
                    all_clists = []
                    all_scores = []
                    min_score, max_score = 114514, -114514
                    worse_index, best_index = -1, -1
                    lock_best = False
                    for i, choice in enumerate(turn_choice_dict, 0):
                        score = choice.get("score")
                        if "best" in choice.keys():
                            best_clist = cList.from_dicts(choice["best"])
                            all_clists.append(best_clist)
                            if score is None:
                                all_scores.append(5)
                                best_index = i
                                max_score = 5
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                best_index = i
                                lock_best = True
                                if score_number > max_score:
                                    max_score = score
                        else:
                            all_clists.append(cList.from_dicts(choice["choice"]))
                            if score is None:
                                all_scores.append(1)
                                worse_index = i
                                min_score = 1
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                if score_number < min_score:
                                    min_score = score
                                    worse_index = i
                                if score_number > max_score:
                                    max_score = score
                                    best_index = i if not lock_best else best_index
                    # 存在best标签则训练。
                    if best_clist is not None:
                        if not allow_multilabel:
                            pos_conversations = best_clist.to_dict_list()
                            neg_conversations = all_clists[worse_index].to_dict_list()
                            pos_tokens, pos_mask = best_clist.to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            neg_tokens, neg_mask = all_clists[worse_index].to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            requests.post(
                                f"{inference_service_server}/infer",
                                json={
                                    "conversations": pos_conversations,
                                    "state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_pos",
                                },
                            )
                            requests.post(
                                f"{inference_service_server}/infer",
                                json={
                                    "conversations": neg_conversations,
                                    "state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_neg",
                                },
                            )
                            logits_pos_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_pos.logits"
                            )
                            logits_neg_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_neg.logits"
                            )
                            logits_ref_p = torch.load(logits_pos_dir)
                            logits_ref_n = torch.load(logits_neg_dir)
                            logits_p, _ = self.model_engine(
                                torch.tensor([pos_tokens], dtype=torch.long), state
                            )
                            logits_n, _ = self.model_engine(
                                torch.tensor([neg_tokens], dtype=torch.long), state
                            )
                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses(
                                    logits_p,
                                    logits_ref_p,
                                    logits_n,
                                    logits_ref_n,
                                    pos_tokens,
                                    neg_tokens,
                                    pos_mask,
                                    neg_mask,
                                    train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()

                            print(
                                f"batch: {step}, conversation:{pos_conversations()} -> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards.item()}, rejected_rewards:{rejected_rewards.item()}"
                            )

                        else:
                            if n_use_max_choices < 0 or n_use_max_choices > len(
                                all_clists
                            ):
                                train_clists = all_clists
                                train_scores = all_scores
                            elif n_use_max_choices <= 2:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                            else:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                                for index in sorted(
                                    [best_index, worse_index], reverse=True
                                ):
                                    del all_scores[index], all_clists[index]
                                assert len(all_clists) == len(all_scores)
                                for i in range(n_use_max_choices - 2):
                                    index = random.randint(0, len(all_clists) - 1)
                                    train_clists.append(all_clists[index])
                                    train_scores.append(all_scores[index])
                                    del all_clists[index], all_scores[index]
                            logits_list = []
                            logits_ref_list = []
                            batch_tokens_list = []
                            batch_masks_list = []

                            for clist in train_clists:
                                tokens, mask = clist.to_tokens(
                                    global_config.tokenizer_train.encode,
                                    use_ego_mask=True,
                                    ego_mask_type="zero",
                                )
                                batch_tokens_list.append(tokens)
                                batch_masks_list.append(mask)
                                requests.post(
                                    f"{inference_service_server}/infer",
                                    json={
                                        "conversations": clist.to_dict_list(),
                                        "state_idx": state_idx,
                                        "save_folder": dpo_cache_folder算法待修改,
                                        "save_name": "dpo",
                                    },
                                )
                                logits_dir = os.path.join(
                                    dpo_cache_folder算法待修改, "dpo.logits"
                                )
                                logits_ref = torch.load(logits_dir)
                                logits_ref_list.append(logits_ref)

                            padded_tokens_list, pad_last_indices = (
                                pad_2d_list_with_zeros(batch_tokens_list)
                            )
                            batch_tokens_tensor = torch.tensor(
                                padded_tokens_list, dtype=torch.long
                            )  # 但是这堆token不一样长
                            batch_logits, _ = self.model_engine(
                                torch.tensor(batch_tokens_tensor, dtype=torch.long),
                                (
                                    state.duplicate(len(batch_tokens_tensor))
                                    if state is not None
                                    else None
                                ),
                            )
                            logits_list = batch_logits.unbind(dim=0)
                            logits_list = [
                                logits[:last_index].unsqueeze(0)
                                for logits, last_index in zip(
                                    logits_list, pad_last_indices
                                )
                            ]

                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses_multilabel(
                                    logits_list,
                                    logits_ref_list,
                                    train_scores,
                                    batch_tokens_list,
                                    batch_masks_list,
                                    beta=train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()
                            gc.collect()
                            torch.cuda.empty_cache()

                            show_index = random.randint(0, len(train_clists) - 1)
                            print(
                                f"batch: {step}, conversation: \n{train_clists[show_index]()}\n, score: {train_scores[show_index]}\n-> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards}, rejected_rewards:{rejected_rewards}"
                            )

                # 加入历史消息
                if "best" in turn_choice_dict[-1].keys():
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["best"])
                else:
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["choice"])
                _, state = self.model_engine(
                    torch.tensor(
                        [hist_clist.to_tokens(global_config.tokenizer_train.encode)[0]],
                        dtype=torch.long,
                    ),
                    state,
                )
                resp = requests.post(
                    f"{inference_service_server}/infer",
                    json={
                        "conversations": hist_clist.to_dict_list(),
                        "state_idx": state_idx,
                        "save_logits": False,
                        "save_to_now_state_idx": state_idx,
                    },
                ).json()

            if step_save_ckpt and step % step_save_ckpt == 0:
                save_path = self.save_weight(
                    f"train_dpo_step:{step}", save_train_state=False
                )
                print(f"====save dpo ckpt at step: {step}, to={save_path}====")

        save_path = self.save_weight(f"train_dpo_final", save_train_state=False)
        print(f"====save dpo ckpt to={save_path}====")

    def train_dpo_v2_iterator(
        self,
        folder_weight_dir_list,
        inference_service_server="http://localhost:4514/",
        step_save_ckpt=None,
        allow_multilabel=True,
        n_use_max_choices=5,
    ):
        dataset = FolderDPODatasetV2(folder_weight_dir_list)

        def test_server(server):
            try:
                response = requests.get(server + "/test")
                if response.status_code == 200:
                    return True
                else:
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Exception occurred: {e}")
                return False

        assert test_server(inference_service_server)

        resp = requests.post(
            f"{inference_service_server}/regist_state_id",
            json={},
        ).json()
        state_idx = resp["access_token"]

        for step, ordered_data_dicts in enumerate(dataset.load_batch_datas(), 0):
            # 每个新对话，重置state
            state = None
            resp = requests.post(
                f"{inference_service_server}/reset_state_id",
                json={"access_token": state_idx},
            ).json()
            for turn_choice_dict in ordered_data_dicts.values():
                if len(turn_choice_dict) > 1:
                    best_clist = None
                    all_clists = []
                    all_scores = []
                    min_score, max_score = 114514, -114514
                    worse_index, best_index = -1, -1
                    lock_best = False
                    for i, choice in enumerate(turn_choice_dict, 0):
                        score = choice.get("score")
                        if "best" in choice.keys():
                            best_clist = cList.from_dicts(choice["best"])
                            all_clists.append(best_clist)
                            if score is None:
                                all_scores.append(5)
                                best_index = i
                                max_score = 5
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                best_index = i
                                lock_best = True
                                if score_number > max_score:
                                    max_score = score
                        else:
                            all_clists.append(cList.from_dicts(choice["choice"]))
                            if score is None:
                                all_scores.append(1)
                                worse_index = i
                                min_score = 1
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                if score_number < min_score:
                                    min_score = score
                                    worse_index = i
                                if score_number > max_score:
                                    max_score = score
                                    best_index = i if not lock_best else best_index
                    # 存在best标签则训练。
                    if best_clist is not None:
                        if not allow_multilabel:
                            pos_conversations = best_clist.to_dict_list()
                            neg_conversations = all_clists[worse_index].to_dict_list()
                            pos_tokens, pos_mask = best_clist.to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            neg_tokens, neg_mask = all_clists[worse_index].to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            requests.post(
                                f"{inference_service_server}/infer",
                                json={
                                    "conversations": pos_conversations,
                                    "state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_pos",
                                },
                            )
                            requests.post(
                                f"{inference_service_server}/infer",
                                json={
                                    "conversations": neg_conversations,
                                    "state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_neg",
                                },
                            )
                            logits_pos_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_pos.logits"
                            )
                            logits_neg_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_neg.logits"
                            )
                            logits_ref_p = torch.load(logits_pos_dir)
                            logits_ref_n = torch.load(logits_neg_dir)
                            logits_p, _ = self.model_engine(
                                torch.tensor([pos_tokens], dtype=torch.long), state
                            )
                            logits_n, _ = self.model_engine(
                                torch.tensor([neg_tokens], dtype=torch.long), state
                            )
                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses(
                                    logits_p,
                                    logits_ref_p,
                                    logits_n,
                                    logits_ref_n,
                                    pos_tokens,
                                    neg_tokens,
                                    pos_mask,
                                    neg_mask,
                                    train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()

                            print(
                                f"batch: {step}, conversation:{pos_conversations()} -> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards.item()}, rejected_rewards:{rejected_rewards.item()}"
                            )
                            yield json.dumps(
                                {
                                    "step": step,
                                    "dpo_loss": dpo_loss.item(),
                                    "chosen_rewards": float(chosen_rewards.mean()),
                                    "rejected_rewards": float(rejected_rewards.mean()),
                                },
                                ensure_ascii=True,
                            ) + "\n"

                        else:
                            if n_use_max_choices < 0 or n_use_max_choices > len(
                                all_clists
                            ):
                                train_clists = all_clists
                                train_scores = all_scores
                            elif n_use_max_choices <= 2:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                            else:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                                for index in sorted(
                                    [best_index, worse_index], reverse=True
                                ):
                                    del all_scores[index], all_clists[index]
                                assert len(all_clists) == len(all_scores)
                                for i in range(n_use_max_choices - 2):
                                    index = random.randint(0, len(all_clists) - 1)
                                    train_clists.append(all_clists[index])
                                    train_scores.append(all_scores[index])
                                    del all_clists[index], all_scores[index]
                            logits_list = []
                            logits_ref_list = []
                            batch_tokens_list = []
                            batch_masks_list = []

                            for clist in train_clists:
                                tokens, mask = clist.to_tokens(
                                    global_config.tokenizer_train.encode,
                                    use_ego_mask=True,
                                    ego_mask_type="zero",
                                )
                                batch_tokens_list.append(tokens)
                                batch_masks_list.append(mask)
                                requests.post(
                                    f"{inference_service_server}/infer",
                                    json={
                                        "conversations": clist.to_dict_list(),
                                        "state_idx": state_idx,
                                        "save_folder": dpo_cache_folder算法待修改,
                                        "save_name": "dpo",
                                    },
                                )
                                logits_dir = os.path.join(
                                    dpo_cache_folder算法待修改, "dpo.logits"
                                )
                                logits_ref = torch.load(logits_dir)
                                logits_ref_list.append(logits_ref)

                            padded_tokens_list, pad_last_indices = (
                                pad_2d_list_with_zeros(batch_tokens_list)
                            )
                            batch_tokens_tensor = torch.tensor(
                                padded_tokens_list, dtype=torch.long
                            )  # 但是这堆token不一样长
                            batch_logits, _ = self.model_engine(
                                torch.tensor(batch_tokens_tensor, dtype=torch.long),
                                (
                                    state.duplicate(len(batch_tokens_tensor))
                                    if state is not None
                                    else None
                                ),
                            )
                            logits_list = batch_logits.unbind(dim=0)
                            logits_list = [
                                logits[:last_index].unsqueeze(0)
                                for logits, last_index in zip(
                                    logits_list, pad_last_indices
                                )
                            ]

                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses_multilabel(
                                    logits_list,
                                    logits_ref_list,
                                    train_scores,
                                    batch_tokens_list,
                                    batch_masks_list,
                                    beta=train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()
                            gc.collect()
                            torch.cuda.empty_cache()

                            show_index = random.randint(0, len(train_clists) - 1)
                            print(
                                f"batch: {step}, conversation: \n{train_clists[show_index]()}\n, score: {train_scores[show_index]}\n-> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards}, rejected_rewards:{rejected_rewards}"
                            )
                            yield json.dumps(
                                {
                                    "step": step,
                                    "dpo_loss": dpo_loss.item(),
                                    "chosen_rewards": float(chosen_rewards.mean()),
                                    "rejected_rewards": float(rejected_rewards.mean()),
                                },
                                ensure_ascii=True,
                            ) + "\n"

                # 加入历史消息
                if "best" in turn_choice_dict[-1].keys():
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["best"])
                else:
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["choice"])
                _, state = self.model_engine(
                    torch.tensor(
                        [hist_clist.to_tokens(global_config.tokenizer_train.encode)[0]],
                        dtype=torch.long,
                    ),
                    state,
                )
                resp = requests.post(
                    f"{inference_service_server}/infer",
                    json={
                        "conversations": hist_clist.to_dict_list(),
                        "state_idx": state_idx,
                        "save_logits": False,
                        "save_to_now_state_idx": state_idx,
                    },
                ).json()

            if step_save_ckpt and step % step_save_ckpt == 0:
                save_path = self.save_weight(
                    f"train_dpo_step:{step}", save_train_state=False
                )
                print(f"====save dpo ckpt at step: {step}, to={save_path}====")

        save_path = self.save_weight(f"train_dpo_final", save_train_state=False)
        print(f"====save dpo ckpt to={save_path}====")
        yield json.dumps(
            {
                "over": True,
                "to_dir": save_path,
            },
            ensure_ascii=True,
        ) + "\n"

    def train_dpo_v3_iterator(
        self,
        folder_weight_dir_list,
        inference_service_server="http://localhost:4514/",
        step_save_ckpt=None,
        allow_multilabel=True,
        n_use_max_choices=5,
    ):
        dataset = FolderDPODatasetV2(folder_weight_dir_list)

        def test_server(server):
            try:
                response = requests.get(server + "/test")
                if response.status_code == 200:
                    return True
                else:
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Exception occurred: {e}")
                return False

        assert test_server(inference_service_server)

        resp = requests.post(
            f"{inference_service_server}/regist_state_id",
            json={},
        ).json()
        state_idx = resp["access_token"]

        hist_tokens = []
        hist_masks = []
        for step, ordered_data_dicts in enumerate(dataset.load_batch_datas(), 0):
            # 每个新对话，重置state
            state = None
            resp = requests.post(
                f"{inference_service_server}/reset_state_id",
                json={"access_token": state_idx},
            ).json()

            for turn_choice_dict in ordered_data_dicts.values():
                if len(turn_choice_dict) > 1:
                    best_clist = None
                    all_clists = []
                    all_scores = []
                    min_score, max_score = 114514, -114514
                    worse_index, best_index = -1, -1
                    lock_best = False
                    for i, choice in enumerate(turn_choice_dict, 0):
                        score = choice.get("score")
                        if "best" in choice.keys():
                            best_clist = cList.from_dicts(choice["best"])
                            all_clists.append(best_clist)
                            if score is None:
                                all_scores.append(5)
                                best_index = i
                                max_score = 5
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                best_index = i
                                lock_best = True
                                if score_number > max_score:
                                    max_score = score
                        else:
                            all_clists.append(cList.from_dicts(choice["choice"]))
                            if score is None:
                                all_scores.append(1)
                                min_score = 1
                                worse_index = i
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                if score_number < min_score:
                                    min_score = score
                                    worse_index = i
                                if score_number > max_score:
                                    max_score = score
                                    best_index = i if not lock_best else best_index
                    # 存在best标签则训练。
                    if best_clist is not None:
                        if not allow_multilabel:
                            pos_tokens, pos_mask = best_clist.to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            neg_tokens, neg_mask = all_clists[worse_index].to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            pos_all_tokens = hist_tokens + pos_tokens
                            pos_all_masks = hist_masks + pos_mask
                            neg_all_tokens = hist_tokens + neg_tokens
                            neg_all_masks = hist_masks + neg_mask

                            resp = requests.post(
                                f"{inference_service_server}/infer_tokenss",
                                json={
                                    "tokens": pos_all_tokens,
                                    "state_idx": state_idx,
                                    "save_to_now_state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_pos",
                                },
                            ).json()
                            resp = requests.post(
                                f"{inference_service_server}/infer_tokenss",
                                json={
                                    "tokens": neg_all_tokens,
                                    "state_idx": state_idx,
                                    "save_to_now_state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_neg",
                                },
                            ).json()
                            logits_pos_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_pos.logits"
                            )
                            logits_neg_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_neg.logits"
                            )
                            logits_ref_p = torch.load(logits_pos_dir)
                            logits_ref_n = torch.load(logits_neg_dir)
                            logits_p, _ = self.model_engine(
                                torch.tensor([pos_all_tokens], dtype=torch.long), state
                            )
                            logits_n, _ = self.model_engine(
                                torch.tensor([neg_all_tokens], dtype=torch.long), state
                            )
                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses(
                                    logits_p,
                                    logits_ref_p,
                                    logits_n,
                                    logits_ref_n,
                                    pos_all_tokens,
                                    neg_all_tokens,
                                    pos_all_masks,
                                    neg_all_masks,
                                    train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()

                            pos_conversations = best_clist.to_dict_list()
                            print(
                                f"batch: {step}, conversation:{pos_conversations()} -> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards.item()}, rejected_rewards:{rejected_rewards.item()}"
                            )
                            yield json.dumps(
                                {
                                    "step": step,
                                    "dpo_loss": dpo_loss.item(),
                                    "chosen_rewards": float(chosen_rewards.mean()),
                                    "rejected_rewards": float(rejected_rewards.mean()),
                                },
                                ensure_ascii=True,
                            ) + "\n"

                        else:
                            if n_use_max_choices < 0 or n_use_max_choices > len(
                                all_clists
                            ):
                                train_clists = all_clists
                                train_scores = all_scores
                            elif n_use_max_choices <= 2:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                            else:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                                for index in sorted(
                                    [best_index, worse_index], reverse=True
                                ):
                                    del all_scores[index], all_clists[index]
                                assert len(all_clists) == len(all_scores)
                                for i in range(n_use_max_choices - 2):
                                    index = random.randint(0, len(all_clists) - 1)
                                    train_clists.append(all_clists[index])
                                    train_scores.append(all_scores[index])
                                    del all_clists[index], all_scores[index]
                            print("train_scores", train_scores)
                            logits_list = []
                            logits_ref_list = []
                            batch_tokens_list = []
                            batch_masks_list = []

                            for clist in train_clists:
                                tokens, mask = clist.to_tokens(
                                    global_config.tokenizer_train.encode,
                                    use_ego_mask=True,
                                    ego_mask_type="zero",
                                )
                                batch_tokens_list.append(hist_tokens + tokens)
                                batch_masks_list.append(hist_masks + mask)

                                resp = requests.post(
                                    f"{inference_service_server}/infer_tokenss",
                                    json={
                                        "tokens": hist_tokens + tokens,
                                        "state_idx": state_idx,
                                        "save_to_now_state_idx": state_idx,
                                        "save_folder": dpo_cache_folder算法待修改,
                                        "save_name": "dpo",
                                    },
                                ).json()

                                logits_dir = os.path.join(
                                    dpo_cache_folder算法待修改, "dpo.logits"
                                )
                                logits_ref = torch.load(logits_dir)
                                logits_ref_list.append(logits_ref)

                            padded_tokens_list, pad_last_indices = (
                                pad_2d_list_with_zeros(batch_tokens_list)
                            )
                            batch_tokens_tensor = torch.tensor(
                                padded_tokens_list, dtype=torch.long
                            )  # 但是这堆token不一样长
                            batch_logits, _ = self.model_engine(
                                torch.tensor(batch_tokens_tensor, dtype=torch.long),
                                (
                                    state.duplicate(len(batch_tokens_tensor))
                                    if state is not None
                                    else None
                                ),
                            )
                            logits_list = batch_logits.unbind(dim=0)
                            logits_list = [
                                logits[:last_index].unsqueeze(0)
                                for logits, last_index in zip(
                                    logits_list, pad_last_indices
                                )
                            ]

                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses_multilabel(
                                    logits_list,
                                    logits_ref_list,
                                    train_scores,
                                    batch_tokens_list,
                                    batch_masks_list,
                                    beta=train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()

                            gc.collect()
                            torch.cuda.empty_cache()

                            show_index = random.randint(0, len(train_clists) - 1)
                            print(
                                f"batch: {step}, conversation: \n{train_clists[show_index]()}\n, score: {train_scores[show_index]}\n-> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards}, rejected_rewards:{rejected_rewards}"
                            )
                            yield json.dumps(
                                {
                                    "step": step,
                                    "dpo_loss": dpo_loss.item(),
                                    "chosen_rewards": float(chosen_rewards.mean()),
                                    "rejected_rewards": float(rejected_rewards.mean()),
                                },
                                ensure_ascii=True,
                            ) + "\n"

                # 加入历史消息
                if "best" in turn_choice_dict[-1].keys():
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["best"])
                else:
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["choice"])

                tt, mm = hist_clist.to_tokens(
                    global_config.tokenizer_train.encode, use_ego_mask=True
                )
                hist_tokens += tt
                hist_masks += mm
                # hist_masks=[0 for _ in hist_tokens]
                threshold = train_config.model.ctx_len // 2
                if len(hist_tokens) > threshold:
                    upload_tokens = hist_tokens[:-threshold]
                    _, state = self.model_engine(
                        torch.tensor(
                            [upload_tokens],
                            dtype=torch.long,
                        ),
                        state,
                    )
                    resp = requests.post(
                        f"{inference_service_server}/infer_tokenss",
                        json={
                            "tokens": upload_tokens,
                            "state_idx": state_idx,
                            "save_to_now_state_idx": state_idx,
                            "save_logits": False,
                        },
                    ).json()
                    hist_tokens = hist_tokens[-threshold:]
                    hist_masks = hist_masks[-threshold:]

            if step_save_ckpt and step % step_save_ckpt == 0:
                save_path = self.save_weight(
                    f"train_dpo_step:{step}", save_train_state=False
                )
                print(f"====save dpo ckpt at step: {step}, to={save_path}====")

        save_path = self.save_weight(f"train_dpo_final", save_train_state=False)
        print(f"====save dpo ckpt to={save_path}====")
        yield json.dumps(
            {
                "over": True,
                "to_dir": save_path,
            },
            ensure_ascii=True,
        ) + "\n"

    def train_dpo_v3(
        self,
        folder_weight_dir_list,
        inference_service_server="http://localhost:4514/",
        step_save_ckpt=None,
        allow_multilabel=True,
        n_use_max_choices=5,
    ):
        dataset = FolderDPODatasetV2(folder_weight_dir_list)

        def test_server(server):
            try:
                response = requests.get(server + "/test")
                if response.status_code == 200:
                    return True
                else:
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Exception occurred: {e}")
                return False

        assert test_server(inference_service_server)

        resp = requests.post(
            f"{inference_service_server}/regist_state_id",
            json={},
        ).json()
        state_idx = resp["access_token"]

        hist_tokens = []
        hist_masks = []
        for step, ordered_data_dicts in enumerate(dataset.load_batch_datas(), 0):
            # 每个新对话，重置state
            state = None
            resp = requests.post(
                f"{inference_service_server}/reset_state_id",
                json={"access_token": state_idx},
            ).json()

            for turn_choice_dict in ordered_data_dicts.values():
                if len(turn_choice_dict) > 1:
                    best_clist = None
                    all_clists = []
                    all_scores = []
                    min_score, max_score = 114514, -114514
                    worse_index, best_index = -1, -1
                    lock_best = False
                    for i, choice in enumerate(turn_choice_dict, 0):
                        score = choice.get("score")
                        if "best" in choice.keys():
                            best_clist = cList.from_dicts(choice["best"])
                            all_clists.append(best_clist)
                            if score is None:
                                all_scores.append(5)
                                best_index = i
                                max_score = 5
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                best_index = i
                                lock_best = True
                                if score_number > max_score:
                                    max_score = score
                        else:
                            all_clists.append(cList.from_dicts(choice["choice"]))
                            if score is None:
                                all_scores.append(1)
                                min_score = 1
                                worse_index = i
                            else:
                                score_number = int(score)
                                all_scores.append(score_number)
                                if score_number < min_score:
                                    min_score = score
                                    worse_index = i
                                if score_number > max_score:
                                    max_score = score
                                    best_index = i if not lock_best else best_index
                    # 存在best标签则训练。
                    if best_clist is not None:
                        if not allow_multilabel:
                            pos_tokens, pos_mask = best_clist.to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            neg_tokens, neg_mask = all_clists[worse_index].to_tokens(
                                global_config.tokenizer_train.encode,
                                use_ego_mask=True,
                                ego_mask_type="zero",
                            )
                            pos_all_tokens = hist_tokens + pos_tokens
                            pos_all_masks = hist_masks + pos_mask
                            neg_all_tokens = hist_tokens + neg_tokens
                            neg_all_masks = hist_masks + neg_mask

                            resp = requests.post(
                                f"{inference_service_server}/infer_tokenss",
                                json={
                                    "tokens": pos_all_tokens,
                                    "state_idx": state_idx,
                                    "save_to_now_state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_pos",
                                },
                            ).json()
                            resp = requests.post(
                                f"{inference_service_server}/infer_tokenss",
                                json={
                                    "tokens": neg_all_tokens,
                                    "state_idx": state_idx,
                                    "save_to_now_state_idx": state_idx,
                                    "save_folder": dpo_cache_folder算法待修改,
                                    "save_name": "dpo_neg",
                                },
                            ).json()
                            logits_pos_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_pos.logits"
                            )
                            logits_neg_dir = os.path.join(
                                dpo_cache_folder算法待修改, "dpo_neg.logits"
                            )
                            logits_ref_p = torch.load(logits_pos_dir)
                            logits_ref_n = torch.load(logits_neg_dir)
                            logits_p, _ = self.model_engine(
                                torch.tensor([pos_all_tokens], dtype=torch.long), state
                            )
                            logits_n, _ = self.model_engine(
                                torch.tensor([neg_all_tokens], dtype=torch.long), state
                            )
                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses(
                                    logits_p,
                                    logits_ref_p,
                                    logits_n,
                                    logits_ref_n,
                                    pos_all_tokens,
                                    neg_all_tokens,
                                    pos_all_masks,
                                    neg_all_masks,
                                    train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()

                            pos_conversations = best_clist.to_dict_list()
                            print(
                                f"batch: {step}, conversation:{pos_conversations()} -> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards.item()}, rejected_rewards:{rejected_rewards.item()}"
                            )

                        else:
                            if n_use_max_choices < 0 or n_use_max_choices > len(
                                all_clists
                            ):
                                train_clists = all_clists
                                train_scores = all_scores
                            elif n_use_max_choices <= 2:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                            else:
                                train_clists = [
                                    all_clists[best_index],
                                    all_clists[worse_index],
                                ]
                                train_scores = [
                                    all_scores[best_index],
                                    all_scores[worse_index],
                                ]
                                for index in sorted(
                                    [best_index, worse_index], reverse=True
                                ):
                                    del all_scores[index], all_clists[index]
                                assert len(all_clists) == len(all_scores)
                                for i in range(n_use_max_choices - 2):
                                    index = random.randint(0, len(all_clists) - 1)
                                    train_clists.append(all_clists[index])
                                    train_scores.append(all_scores[index])
                                    del all_clists[index], all_scores[index]
                            print("train_scores", train_scores)
                            logits_list = []
                            logits_ref_list = []
                            batch_tokens_list = []
                            batch_masks_list = []

                            for clist in train_clists:
                                tokens, mask = clist.to_tokens(
                                    global_config.tokenizer_train.encode,
                                    use_ego_mask=True,
                                    ego_mask_type="zero",
                                )
                                batch_tokens_list.append(hist_tokens + tokens)
                                batch_masks_list.append(hist_masks + mask)

                                resp = requests.post(
                                    f"{inference_service_server}/infer_tokenss",
                                    json={
                                        "tokens": hist_tokens + tokens,
                                        "state_idx": state_idx,
                                        "save_to_now_state_idx": state_idx,
                                        "save_folder": dpo_cache_folder算法待修改,
                                        "save_name": "dpo",
                                    },
                                ).json()

                                logits_dir = os.path.join(
                                    dpo_cache_folder算法待修改, "dpo.logits"
                                )
                                logits_ref = torch.load(logits_dir)
                                logits_ref_list.append(logits_ref)

                            padded_tokens_list, pad_last_indices = (
                                pad_2d_list_with_zeros(batch_tokens_list)
                            )
                            batch_tokens_tensor = torch.tensor(
                                padded_tokens_list, dtype=torch.long
                            )  # 但是这堆token不一样长
                            batch_logits, _ = self.model_engine(
                                torch.tensor(batch_tokens_tensor, dtype=torch.long),
                                (
                                    state.duplicate(len(batch_tokens_tensor))
                                    if state is not None
                                    else None
                                ),
                            )
                            logits_list = batch_logits.unbind(dim=0)
                            logits_list = [
                                logits[:last_index].unsqueeze(0)
                                for logits, last_index in zip(
                                    logits_list, pad_last_indices
                                )
                            ]

                            dpo_loss, chosen_rewards, rejected_rewards = (
                                self.calc_dpo_losses_multilabel(
                                    logits_list,
                                    logits_ref_list,
                                    train_scores,
                                    batch_tokens_list,
                                    batch_masks_list,
                                    beta=train_config.dpo.beta,
                                )
                            )
                            self.model_engine.backward(dpo_loss)
                            self.model_engine.step()

                            gc.collect()
                            torch.cuda.empty_cache()

                            show_index = random.randint(0, len(train_clists) - 1)
                            print(
                                f"batch: {step}, conversation: \n{train_clists[show_index]()}\n, score: {train_scores[show_index]}\n-> loss:{dpo_loss.item()}, chosen_rewards:{chosen_rewards}, rejected_rewards:{rejected_rewards}"
                            )

                # 加入历史消息
                if "best" in turn_choice_dict[-1].keys():
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["best"])
                else:
                    hist_clist = cList.from_dicts(turn_choice_dict[-1]["choice"])

                tt, mm = hist_clist.to_tokens(
                    global_config.tokenizer_train.encode, use_ego_mask=True
                )
                hist_tokens += tt
                hist_masks += mm
                # hist_masks=[0 for _ in hist_tokens]
                threshold = train_config.model.ctx_len // 2
                if len(hist_tokens) > threshold:
                    upload_tokens = hist_tokens[:-threshold]
                    _, state = self.model_engine(
                        torch.tensor(
                            [upload_tokens],
                            dtype=torch.long,
                        ),
                        state,
                    )
                    resp = requests.post(
                        f"{inference_service_server}/infer_tokenss",
                        json={
                            "tokens": upload_tokens,
                            "state_idx": state_idx,
                            "save_to_now_state_idx": state_idx,
                            "save_logits": False,
                        },
                    ).json()
                    hist_tokens = hist_tokens[-threshold:]
                    hist_masks = hist_masks[-threshold:]

            if step_save_ckpt and step % step_save_ckpt == 0:
                save_path = self.save_weight(
                    f"train_dpo_step:{step}", save_train_state=False
                )
                print(f"====save dpo ckpt at step: {step}, to={save_path}====")

        save_path = self.save_weight(f"train_dpo_final", save_train_state=False)
        print(f"====save dpo ckpt to={save_path}====")

    def build_engine(self):
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
                lr=self.args.train.lr_init,
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
                lr=self.args.train.lr_init,
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
            warmup_min_lr=self.args.train.lr_init,
            warmup_max_lr=self.args.train.lr_final,
            warmup_num_steps=self.args.train.warmup_steps,
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
