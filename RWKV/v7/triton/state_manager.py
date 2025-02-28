import torch
from RWKV.v7.triton.state import BlockStateList, BlockState, TimeMixState, ChannelMixState

class RWKVStateManager:
    def __init__(self, model, batch_size, device, dtype=torch.float16):
        self.model = model
        self.n_layer = model.n_layer
        self.n_embd = model.n_embd
        self.head_size = model.head_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # Get number of heads
        n_head = self.n_embd // self.head_size
        
        # Initialize states
        self.states = BlockStateList.create(
            self.n_layer, 
            self.batch_size, 
            self.n_embd, 
            n_head, 
            self.device, 
            self.dtype
        )
        
    def reset_states(self):
        """Reset all states to zero"""
        n_head = self.n_embd // self.head_size
        self.states = BlockStateList.create(
            self.n_layer, 
            self.batch_size, 
            self.n_embd, 
            n_head, 
            self.device, 
            self.dtype
        )
    
    def detach_states(self):
        """Detach states from computation graph (for truncated BPTT)"""
        self.states.wkv_states = self.states.wkv_states.detach()
        self.states.shift_states = self.states.shift_states.detach()
    
    def get_layer_state(self, layer_idx):
        """Get the state for a specific layer"""
        return self.states[layer_idx]
        
    def update_layer_state(self, layer_idx, new_state):
        """Update the state for a specific layer"""
        self.states[layer_idx] = new_state 