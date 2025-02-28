import torch.nn as nn
from RWKV.v7.triton.time_mix import RWKV_Tmix_x070
from RWKV.v7.triton.channel_mix import RWKV_CMix_x070
from RWKV.v7.triton.state import BlockState

class Block(nn.Module):
    def __init__(self, 
                 n_embd, 
                 n_layer, 
                 layer_id, 
                 dim_att=None, 
                 head_size=64, 
                 head_size_divisor=8, 
                 dropout=0.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.dim_att = dim_att if dim_att is not None else n_embd
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_Tmix_x070(n_embd=n_embd, 
                                  n_layer=n_layer, 
                                  layer_id=layer_id, 
                                  dim_att=self.dim_att, 
                                  head_size=head_size, 
                                  head_size_divisor=head_size_divisor)
                                  
        self.ffn = RWKV_CMix_x070(n_embd=n_embd, 
                                 n_layer=n_layer, 
                                 layer_id=layer_id)

        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)

    def forward(self, x, v_first, time_mix_state, channel_mix_state):
        """Forward pass with explicit state management
        
        Args:
            x: Input tensor [B, T, C]
            v_first: First layer's value or None
            time_mix_state: Optional state for time mixing
            channel_mix_state: Optional state for channel mixing
            
        Returns:
            x: Output tensor
            v_first: Updated v_first
            new_time_state: New time mixing state
            new_channel_state: New channel mixing state
        """
        if self.layer_id == 0:
            x = self.ln0(x)

        # Time mix with state
        x_ln1 = self.ln1(x)
        x_attn, v_first, new_time_state = self.att(x_ln1, v_first, time_mix_state)
        
        if self.dropout > 0:
            x_attn = self.drop0(x_attn)
            
        x = x + x_attn

        # Channel mix with state
        x_ln2 = self.ln2(x)
        ffn_out, new_channel_state = self.ffn(x_ln2, channel_mix_state)
        
        if self.dropout > 0:
            ffn_out = self.drop1(ffn_out)
            
        x = x + ffn_out

        return x, v_first, new_time_state, new_channel_state
