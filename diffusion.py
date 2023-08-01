import torch.nn.functional as F

import torch.nn as nn
import torch


class Diffusion:
    def __init__(self, model:   nn.Module,
                       n_steps: int          = 1000,
                       betas:   torch.Tensor = None,
                       criterion             = F.mse_loss) -> None:
        
        assert len(betas.shape) == 1
        assert betas.shape[0]   == n_steps

        self.model   = model
        self.n_steps = n_steps

        self.betas  = betas
        self.alphas = 1 - betas

        self.criterion = criterion

        if self.betas is None:
            self.betas = torch.linspace(0.0001, 0.04, (n_steps,))
    
    def forward(self, x0):
        self.model.train()

        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=x0.device)

        x      = self.forward_diffusion_t_batch(x, t)
        x, eps = self.forward_diffusion_batch(x, t)

        out = self.model(x)

        loss = self.criterion(out, eps)

        return loss

    def forward_diffusion_batch(self, x: torch.Tensor, t: torch.Tensor):
        r, eps = [], []

        for i in range(x.shape[0]):
            r_, eps_ = self.forward_diffusion(x[i], t[i])
            r.append(r_[None]); eps.append(eps_[None])

        r   = torch.cat(r,   dim=0)
        eps = torch.cat(eps, dim=0)

        return r, eps
    
    def forward_diffusion_t_batch(self, x: torch.Tensor, t: torch.Tensor):
        r = []

        for i in range(x.shape[0]):
            r.append(self.forward_diffusion_t(x[i], t[i])[None])
        
        r = torch.cat(r, dim=0)

        return r

    def forward_diffusion(self, x: torch.Tensor, t: int):
        eps = torch.randn_like(x)
        alpha = self.alphas[t]

        return x * (alpha ** 0.5) + eps * ((1 - alpha) ** 0.5), eps

    def forward_diffusion_t(self, x0: torch.Tensor, t: int):
        assert t <= self.n_steps

        if t == 0:
            return x0
        
        eps = torch.randn_like(x0)
        A   = torch.prod(self.alphas[:t])

        r   = torch.sqrt(A) * x0 + torch.sqrt(1 - A) * eps

        return r
    
    def denoise(self, x: torch.Tensor, eps: torch.Tensor, alpha: int = 1):
        return (x - eps * torch.sqrt(1 - alpha)) / (alpha ** 0.5)
    
    def denoise_batch(self, x: torch.Tensor, eps: torch.Tensor, alpha: int):
        r = []

        for i in range(x.shape[0]):
            r.append(self.denoise(x[i], eps[i], alpha)[None])
        
        r = torch.cat(r, dim=0)

        return r
    
    @torch.no_grad()
    def sample(self, n, input_shape=(3, 64, 64), device="cpu"):
        self.model.eval()
        xt = torch.randn((n, *input_shape), device=device)

        for i in range(0, self.n_steps)[::-1]:
            t   = torch.full((n,), i, device=device, dtype=torch.long)

            eps = self.model(xt, t)
            xt  = self.denoise_batch(xt, eps, self.alphas[i])

        self.model.train()
        return xt
    
    @torch.no_grad()
    def evalute(self, c=10, input_shape=(3, 64, 64), device="cpu"):
        self.model.eval()

        xt = torch.randn((1, *input_shape), device=device)
        stepsize = int(self.n_steps / c)
        xr = []
        
        for i in range(0, self.n_steps)[::-1]:
            t   = torch.full((1,), i, device=device, dtype=torch.long)

            eps = self.model(xt, t)
            xt  = self.denoise_batch(xt, eps, self.alphas[i])
            
            if i % stepsize == 0:
                xr.append(xt.clone().detach())

        self.model.train()
        xr = torch.cat(xr, dim=0)
        
        return xr