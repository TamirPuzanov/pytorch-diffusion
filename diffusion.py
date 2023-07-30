import torch.nn as nn
import torch

import torch.nn.functional as F


def gather(vals, t, x_shape):
    out = vals.gather(-1, t.cpu())
    out = out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
    return out.to(t.device)


class Diffusion(nn.Module):
    def __init__(self, model: nn.Module, criterion=nn.MSELoss(reduction="mean"),
                       n_steps: int = 1000, betas: torch.Tensor = None):
        
        super().__init__()
        
        if betas is None:
            betas = torch.linspace(0.0001, 0.04, n_steps)
        
        assert isinstance(betas, torch.Tensor)
        assert len(betas.shape) == 1
        assert betas.shape[0]   == n_steps
        assert n_steps          >= 1
        
        self.model     = model
        self.criterion = criterion
        
        self.n_steps = n_steps
        self.betas   = betas
        self.calc()
        
    def forward(self, x0: torch.Tensor):
        self.model.train()
        t = torch.randint(
            0, self.n_steps, (x0.shape[0],), 
            dtype=torch.long, device=x0.device
        )
        xn, eps  = self.forward_diffusion_sample(x0, t, x0.device)
        eps_pred = self.model(x=xn, t=t)
        loss = self.criterion(eps_pred, eps)
        return loss
    
    def calc(self):
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def forward_diffusion_sample(self, x_0: torch.Tensor, t: torch.Tensor, device="cpu"):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = gather(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    def sample_timestep(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        betas_t = gather(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = gather(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = gather(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = gather(self.posterior_variance, t, x.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, n, input_shape=(3, 64, 64), device="cpu"):
        self.model.eval()
        xt = torch.randn((n, *input_shape), device=device)
        for i in range(0, self.n_steps)[::-1]:
            t   = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.model(xt, t)
            xt  = self.sample_timestep(xt, eps, t)
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
            xt  = self.sample_timestep(xt, eps, t)
            
            if i % stepsize == 0:
                xr.append(xt.clone().detach())
        self.model.train()
        xr = torch.cat(xr, dim=0)
        
        return xr