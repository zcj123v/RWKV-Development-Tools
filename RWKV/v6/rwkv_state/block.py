import torch
import torch.nn as nn
from .time_mix import TimeMix
from .channel_mix import ChannelMix


from torch.nn import functional as F
from torch.nn import functional
import math
import copy


class EncoderDecoderLora(nn.Module):
    def __init__(self, head_size, emb, r=8, dropout=0.01, alpha=32):
        super().__init__()
        self.encode_weight = nn.Parameter(torch.empty((head_size, head_size)))
        self.encode = nn.Parameter(torch.empty(r, head_size))
        self.decode = nn.Parameter(torch.empty(head_size, r))
        self.encode_dropout = nn.Dropout(dropout)
        self.encode_ln = nn.LayerNorm(head_size)
        self.scaling = alpha / r

        # 用于融合 att_shift 信息的全连接层
        nn.init.kaiming_uniform_(self.encode_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.encode, a=math.sqrt(5))
        nn.init.zeros_(self.decode)

    def forward(self, att_state, ffn_shift):
        # x.shape (1 40 64 64)
        # att_shift (1, 2560)
        x = att_state[1]
        att_shift = att_state[0]
        x = x.to(dtype=torch.bfloat16)
        x = functional.linear(x, self.encode_weight) + self.scaling * functional.linear(
            functional.linear(self.encode_dropout(x), self.encode), self.decode
        )
        # output (1 40 64 64)
        return (att_shift, x.float()), ffn_shift


class EncoderDecoder(nn.Module):
    def __init__(self, head_size, emb, r=64):
        super().__init__()
        self.encode = nn.Linear(r, head_size, bias=False)
        self.encode.weight.data = torch.eye(r, head_size)
        # self.encode_decode_middle =  nn.Linear(r, r, bias=False)
        # self.encode_decode_middle.data =  torch.eye(r, r)
        # self.decode = nn.Linear(head_size, r, bias=False)
        # self.decode.weight.data = torch.eye(head_size, r)
        self.encode_ln = nn.LayerNorm(head_size)
        self.encode_dropout = nn.Dropout(0.05)
        # 用于融合 att_shift 信息的全连接层

    def forward(self, att_state, ffn_shift):
        # x.shape (1 40 64 64)
        # att_shift (1, 2560)
        x = att_state[1]
        att_shift = att_state[0]

        if self.training:
            x = self.encode_dropout(x)

        x = x.to(dtype=torch.bfloat16)
        # x = self.encode_ln(x)
        x = self.encode(x)
        # x = self.encode_decode_middle(x)
        # x = self.decode(x)
        # output (1 40 64 64)
        return (att_shift, x.float()), ffn_shift


class BlockState:
    def __init__(
        self,
        time_mix_state: tuple[torch.Tensor, torch.Tensor],
        channel_mix_state: torch.Tensor,
    ):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    def __init__(self, shift_states, wkv_states):
        self.shift_states = shift_states
        self.wkv_states = wkv_states

    def clone(self):
        return BlockStateList(
            self.shift_states.detach().clone(), self.wkv_states.detach().clone()
        )

    @torch.no_grad()
    def duplicate(self, times):
        wkv_states_list = []
        shift_states_list = []
        for _ in range(times):
            wkv_states_list.append(copy.deepcopy(self.wkv_states))
            shift_states_list.append(copy.deepcopy(self.shift_states))
        return BlockStateList(
            shift_states=torch.cat(shift_states_list, dim=2),
            wkv_states=torch.cat(wkv_states_list, dim=1),
        )

    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        wkv_states = torch.empty(
            (N, B, n_head, head_size, head_size), device=device, dtype=torch.float
        )
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    @staticmethod
    def merge(s1, s2, s1_part: float = 0.5, s2_part: float = 0.5):
        wkv_states = s1.wkv_states * s1_part + s2.wkv_states * s2_part
        shift_states = s1.shift_states * s1.s1_part + s2.shift_states * s2_part
        return BlockStateList(shift_states, wkv_states)

    def decay(self, ratio: float = 0.95):
        if ratio == 0:
            return self
        self.wkv_states = self.wkv_states * ratio
        self.shift_states = self.shift_states
        return self

    def __getitem__(self, layer: int):

        if isinstance(layer, int):  # 保持原有的整数索引功能
            return BlockState(
                (self.shift_states[layer, 0], self.wkv_states[layer]),
                (self.shift_states[layer, 1]),
            )
        elif isinstance(layer, slice):  # 新增切片功能
            new_wkv_states = self.wkv_states[layer]
            new_shift_states = self.shift_states[layer]
            return BlockStateList(new_shift_states, new_wkv_states)
        else:
            raise TypeError(f"Invalid index type: {type(layer)}")

        # return BlockState(
        #     (self.shift_states[layer, 0], self.wkv_states[layer]),
        #     (self.shift_states[layer, 1]),
        # )

    def __add__(self, other):
        if other is None:
            N, B, n_head, head_size, head_size = self.wkv_states.size()
            _, _, _, C = self.shift_states.size()
            return self.__add__(
                BlockStateList.create(
                    N,
                    B,
                    C,
                    n_head,
                    head_size,
                    self.wkv_states.device,
                    self.wkv_states.dtype,
                )
            )
        assert isinstance(other, BlockStateList)
        wkv_states = torch.cat([self.wkv_states, other.wkv_states], dim=1)
        shift_states = torch.cat([self.shift_states, other.shift_states], dim=2)
        return BlockStateList(shift_states, wkv_states)

    def remove_at(self, batch_idx):
        if not (0 <= batch_idx < self.wkv_states.size(1)):
            raise ValueError(
                f"dim_index 超出范围，应在 [0, batch_size: {self.wkv_states.size(1) - 1}] 之间。"
            )
        new_wkv_states = torch.cat(
            (
                self.wkv_states[:, :batch_idx, :, :, :],
                self.wkv_states[:, batch_idx + 1 :, :, :, :],
            ),
            dim=1,
        )
        new_shift_states = torch.cat(
            (
                self.shift_states[:, :, :batch_idx, :],
                self.shift_states[:, :, batch_idx + 1 :, :],
            ),
            dim=2,
        )
        return BlockStateList(new_shift_states, new_wkv_states)

    def unbind(self):
        _wkv = [x.unsqueeze(1) for x in torch.unbind(self.wkv_states, dim=1)]
        _shift = [x.unsqueeze(2) for x in torch.unbind(self.shift_states, dim=2)]
        res = [
            BlockStateList(shift_states, wkv_states)
            for wkv_states, shift_states in zip(_wkv, _shift)
        ]
        return res

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state

    def cpu(self):
        wkv_states = self.wkv_states.detach().cpu()
        shift_states = self.shift_states.detach().cpu()
        return BlockStateList(shift_states, wkv_states)

    def cuda(self):
        wkv_states = self.wkv_states.to("cuda")
        shift_states = self.shift_states.to("cuda")
        return BlockStateList(shift_states, wkv_states)

    def to(self, device):
        wkv_states = self.wkv_states.to(device)
        shift_states = self.shift_states.to(device)
        return BlockStateList(shift_states, wkv_states)

    @classmethod
    def save(cls, item, path):
        item = item.cpu()
        data = {"wkv_states": item.wkv_states, "shift_states": item.shift_states}
        torch.save(data, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu")
        wkv_states = data["wkv_states"]
        shift_states = data["shift_states"]
        item = cls(shift_states, wkv_states)
        return item


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        # add state encoder decoder
        if self.args.lora.train_state:
            self.state_encoder = EncoderDecoder(self.args.head_size, self.args.n_embd)
            # self.state_encoder = LSTMModelEncoder()

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix(args, layer_id)

        self.ffn = ChannelMix(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)

        # state encoder
        if self.args.lora.train_state:
            att_state, ffn_state = self.state_encoder(
                last_state.time_mix_state, last_state.channel_mix_state
            )
        else:
            att_state = last_state.time_mix_state
            ffn_state = last_state.channel_mix_state
        # print(next(self.parameters()).dtype,"<----<")
        # print(next(self.ln1.parameters()).dtype,"<<<<<")
        # print(x.dtype,">>>>>") #为什么这里是float
        att_out, att_state = self.att(self.ln1(x), att_state)

        if self.args.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, ffn_state = self.ffn(self.ln2(x), ffn_state)
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, ffn_state = self.ffn(self.ln2(x), ffn_state)
            x = x + ffn_out

        return x, BlockState(att_state, ffn_state)
