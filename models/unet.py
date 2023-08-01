import torch.nn.functional as F

import torch.nn as nn
import torch

import math

from typing import List


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int, n_steps: int = 1000):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BasicBlock(nn.Module):
    def __init__(self, input_c: int, out_c: int) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_c, out_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(out_c),

            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(out_c),
        )
    
    def forward(self, x):
        return self.main(x)
    

class UpsampleBlock(nn.Module):
    def __init__(self, input_c: int, out_c: int) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_c, out_c, 4, stride=2, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(out_c),
        )
    
    def forward(self, x):
        return self.main(x)
    

class Down(nn.Module):
    def __init__(self, input_c: int, out_c: int, time_emb_dim: int = 32,
                 block = BasicBlock, pool = F.max_pool2d) -> None:
        
        super().__init__()

        self.main = block(input_c, out_c)
        self.pool = pool

        self.fc = nn.Sequential(
            nn.Linear(time_emb_dim, input_c * 2),
            nn.SiLU(), nn.LayerNorm(input_c * 2),

            nn.Linear(input_c * 2, input_c),
            nn.SiLU(), nn.LayerNorm(input_c),
        )
    
    def forward(self, x, t):
        t = self.fc(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        skip = self.main(x + t)
        x    = self.pool(skip, kernel_size=(2, 2))

        return x, skip
    

class Up(nn.Module):
    def __init__(self, input_c: int, skip_c: int, out_c: int, time_emb_dim: int = 32,
                       up_sample = UpsampleBlock, block = BasicBlock) -> None:
        
        super().__init__()

        self.up_sample = up_sample(input_c, input_c)
        self.block     = block(input_c + skip_c, out_c)

        self.fc = nn.Sequential(
            nn.Linear(time_emb_dim, input_c * 2),
            nn.SiLU(), nn.LayerNorm(input_c * 2),

            nn.Linear(input_c * 2, input_c),
            nn.SiLU(), nn.LayerNorm(input_c),
        )
    
    def forward(self, x, skip, t):
        t = self.fc(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = self.up_sample(x + t)

        x = torch.cat((x, skip), dim=1)
        x = self.block(x)

        return x


class Unet(nn.Module):
    def __init__(self, input_c:  int, 
                       c:        List[int] = [32, 64, 128, 256], 
                       ls_c:     int       = 512, 
                       up_sample           = UpsampleBlock, 
                       block               = BasicBlock, 
                       pool                = F.max_pool2d,
                       time_encoder        = SinusoidalPositionEmbeddings,
                       time_emb_dim        = 32,
                       n_steps             = 1000,
                       fn                  = None ) -> None:
        
        super().__init__()

        c.insert(0, input_c)
        
        self.time_encoder = time_encoder(dim=time_emb_dim, n_steps=n_steps)

        self.down_layers = nn.ModuleList([
            Down(c[i], c[i + 1], block=block, pool=pool, time_emb_dim=time_emb_dim)
            for i in range(len(c) - 1)
        ])

        self.up_layers   = nn.ModuleList([
            Up(c[i + 1], c[i + 1], c[i], up_sample=up_sample, block=block, time_emb_dim=time_emb_dim)
            for i in range(len(c) - 1)[::-1]
        ])

        self.ls_conv = nn.Sequential(
            nn.Conv2d(c[-1], ls_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(ls_c),

            nn.Conv2d(ls_c, c[-1], 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(c[-1]),
        )

        self.out_c = nn.Sequential(
            nn.Conv2d(c[0], input_c, 1, stride=1, padding=0),
        )

        self.fn = fn
    
    def forward(self, x, t):
        t = self.time_encoder(t)
        s = []

        for layer in self.down_layers:
            x, skip = layer(x, t)
            s.append(skip)
        
        x = self.ls_conv(x)

        for layer, skip in zip(self.up_layers, s[::-1]):
            x = layer(x, skip, t)
        
        x = self.out_c(x)
        
        if self.fn is not None:
            x = self.fn(x)

        return x