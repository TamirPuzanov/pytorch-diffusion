import torch.nn.functional as F

import torch.nn as nn
import torch

from .position_embeddings import *

from typing import List


class BasicBlock(nn.Module):
    def __init__(self, input_c: int, out_c: int) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_c, out_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(out_c),

            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(out_c),

            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
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

            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(out_c),
        )
    
    def forward(self, x):
        return self.main(x)
    

class Down(nn.Module):
    def __init__(self, input_c: int, out_c: int, time_emb_c: int = 32, 
                 block = BasicBlock, pool = F.max_pool2d) -> None:
        
        super().__init__()

        self.main = block(input_c + time_emb_c, out_c)
        self.pool = pool
    
    def forward(self, x, emb):
        skip = self.main(torch.cat((x, emb), dim=1))
        x    = self.pool(skip, kernel_size=(2, 2))

        return x, skip
    

class Up(nn.Module):
    def __init__(self, input_c: int, skip_c: int, out_c: int, time_emb_c: int = 32, 
                       up_sample = UpsampleBlock, block = BasicBlock) -> None:
        
        super().__init__()

        self.up_sample = up_sample(input_c, input_c)
        self.block     = block(input_c + skip_c + time_emb_c, out_c)
    
    def forward(self, x, skip, time_emb):
        x = self.up_sample(x)

        x = torch.cat((x, skip, time_emb), dim=1)
        x = self.block(x)

        return x
    

class TimeEncoder(nn.Module):
    def __init__(self, input_c: int = 16, emb_c: int = 32, n_layers: int = 4) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_c, emb_c, 3, stride=1, padding=1),
            nn.LeakyReLU(), nn.BatchNorm2d(emb_c)
        )

        self.main = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(emb_c, emb_c, 4, stride=2, padding=1),
                nn.LeakyReLU(), nn.BatchNorm2d(emb_c),

                nn.Conv2d(emb_c, emb_c, 3, stride=1, padding=1),
                nn.LeakyReLU(), nn.BatchNorm2d(emb_c),

                nn.Conv2d(emb_c, emb_c, 3, stride=1, padding=1),
                nn.LeakyReLU(), nn.BatchNorm2d(emb_c)
            )

            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        x   = self.conv(x)
        emb = []

        for layer in self.main:
            x = layer(x)
            emb.append(x)

        return emb 


class Unet(nn.Module):
    def __init__(self, input_c:    int, 
                       c:          List[int] = [32, 64, 128, 256], 
                       ls_c:       int       = 512, 
                       input_size: int       = 64,
                       up_sample             = UpsampleBlock, 
                       block                 = BasicBlock, 
                       pool                  = F.max_pool2d,
                       time_emb              = SinusoidalPositionEmbeddings,
                       time_encoder          = TimeEncoder,
                       time_encoder_c        = 64,
                       fn                    = lambda x: x ) -> None:
        
        super().__init__()

        c.insert(0, input_c)

        assert input_size / (2 ** (len(c) - 1)) == 4

        self.down_layers = nn.ModuleList([
            Down(c[i], c[i + 1], block=block, pool=pool)
            for i in range(len(c) - 1)
        ])

        self.up_layers   = nn.ModuleList([
            Up(c[i + 1], c[i + 1], c[i], up_sample=up_sample, block=block)
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

        self.time_emb = nn.Sequential(
            time_emb(2 * 4 * 4),

            nn.Linear(2 * 4 * 4, 16 * 4 * 4), nn.SiLU(),
            nn.Unflatten(1, (16, 4, 4))
        )

        self.time_encoder = time_encoder(16, time_encoder_c, n_layers=len(c) - 1)

        self.fn = fn
    
    def forward(self, x, t):
        time_emb = self.time_encoder(self.time_emb(t))
        s = []

        for layer, emb in zip(self.down_layers, time_emb[::-1]):
            x, skip = layer(x, emb)
            s.append(skip)
        
        x = self.ls_conv(x)

        for layer, skip, emb in zip(self.up_layers, s[::-1], time_emb):
            x = layer(x, skip, emb)
        
        x = self.out_c(x)
        x = self.fn(x)

        return x
