import torch
import torch.nn as nn
from RWKV.v7.triton.state import ChannelMixState

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop

class RWKV_CMix_x070(MyModule):
    def __init__(self, n_embd, n_layer, layer_id):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))


        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(n_embd**0.5), 0.5/(n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x, last_state: ChannelMixState):
        #xx = self.time_shift(x) - x
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :0]), dim=1) - x
        # print(f'xx dtype = {xx.dtype}')
        # print(f'x dtype = {x.dtype}')
        # print(f'last_state.shift_state dtype = {last_state.shift_state.dtype}')
        # print(f'self.x_k  dtype = {self.x_k.dtype}')
        k = x + xx * self.x_k
        k = torch.relu(self.key(k.to(torch.bfloat16))) ** 2

        return self.value(k), ChannelMixState(x[:, -1])
    
