import torch
######state
class TimeMixState:
    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:
    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:
    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, H, device, dtype):
        result = BlockStateList.empty(N, B, C, H, device, dtype)
        result.wkv_states[:] = 1e-18
        result.shift_states[:] = 1e-18
        return result

    @staticmethod
    def empty(N, B, C, H, device, dtype):
        wkv_states = torch.empty((N, B, H, C//H, C//H),
                                 device=device,
                                 dtype=torch.float32)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

    def to_cuda(self):
        self.wkv_states = self.wkv_states.cuda()
        self.shift_states = self.shift_states.cuda()
        return self

    def to_cpu(self):
        self.wkv_states = self.wkv_states.detach().cpu()
        self.shift_states = self.shift_states.detach().cpu()
        return self

    def save_state_to_path(self, path: str):
        """Save state tensors to file using torch.save"""
        torch.save({
            "wkv": self.wkv_states,
            "shift": self.shift_states
        }, path)

    def load_state_from_path(self, path: str):
        """Load state tensors from file using torch.load"""
        checkpoint = torch.load(path, map_location='cpu')
        self.wkv_states = checkpoint['wkv'].to(self.wkv_states.device)
        self.shift_states = checkpoint['shift'].to(self.shift_states.device)
        return self
