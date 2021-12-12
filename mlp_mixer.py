import torch
from torch import nn
import random
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, out_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, out_dims)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class Mixer(nn.Module):
    def __init__(self, num_patches, input_channels, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.token_mlp = MLP(num_patches, token_mlp_dim, num_patches)
        self.channel_mlp = MLP(input_channels, channel_mlp_dim, input_channels)
        self.layernorm1 = nn.LayerNorm(input_channels, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(input_channels, eps=1e-6)

    def forward(self, x):
        """
        x = (BatchSize, PatchCount, Channels)
        """
        y = self.layernorm1(x)
        y = torch.transpose(y, 1, 2)
        y = self.token_mlp(y)
        y = torch.transpose(y, 1, 2)
        z = y + x
        y = self.layernorm2(z)
        return  z + self.channel_mlp(y)

class MLPMixer(nn.Module):
    def __init__(self, input_dims, # (C,H,W)
                       patch_size, # Int
                       num_classes, # Target classes
                       num_blocks, # Mixer blocks
                       hidden_dims, # How many channels each patch should be mapped at stem
                       token_mlp_dim, # Hidden dimensions for token MLP
                       channel_mlp_dim, # Hidden dimensions for channel MLP
                       stochastic_depth_p = 0, # Probability for Stochastic Depth (https://arxiv.org/abs/1603.09382)
                       dropout_p = 0): # Probability for dropout before the final classifier layer
        super().__init__()
        C, H, W = input_dims
        assert (H*W) % (patch_size**2) == 0 # Make sure equal patch sizes are possible
        self.num_patches = (H*W) // (patch_size**2)
        self.stem = nn.Conv2d(C, hidden_dims, kernel_size=patch_size, stride=patch_size)
        self.mixers = nn.Sequential(*[Mixer(self.num_patches, hidden_dims, token_mlp_dim, channel_mlp_dim) for _ in range(num_blocks)])
        self.layernorm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.classifier = nn.Linear(hidden_dims, num_classes)
        nn.init.zeros_(self.classifier.weight)
        self.use_stochastic_depth = stochastic_depth_p > 0
        self.stochastic_depth_p = np.linspace(0, stochastic_depth_p, num_blocks) # Probability values (linearly increasing) for Stochastic Depth
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self, x):
        y = self.stem(x) # y is now N C H W
        y = torch.flatten(y, start_dim=2, end_dim=3) # y is now N C P
        y = torch.transpose(y, 1, 2) # y is now N P C
        if self.training and self.use_stochastic_depth:
            for i, mixer_block in enumerate(self.mixers):
                if random.random() < self.stochastic_depth_p[i]: # Drop the block
                    pass # y = id(y)
                else:
                    y = mixer_block(y) 
        else: y = self.mixers(y)
        y = self.layernorm(y)
        y = torch.mean(y, dim=1, keepdim=False)
        y = self.dropout(y)
        y = self.classifier(y)
        return y