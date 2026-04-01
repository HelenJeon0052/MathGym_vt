import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------
# EDM / Karras preconditioning : x_pred
# ---------------------------------------

class EMPPre(nn.Module):
    """
    F_theta > x0_hat
    expected
        - input : (x_in, cond)
        - its shape : correspond with image
    """
    
    def __init__(self, net:nn.Module, sigma_data: float):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_a = self.sigma_data
        sigma_b = sigma ** 2
        sigma_aa = sigma_a ** 2
        
        c_in = 1.0 / torch.sqrt(sigma_b + sigma_aa)
        c_skip = sigma_a ** 2 / (sigma_b + sigma_aa)
        c_out = sigma * sigma_a / torch.sqrt(sigma_b + sigma_aa)
        c_noise = torch.log(sigma)
        print(f'c_noise shape match with cond_dim of vit: {c_noise.shape}')
        c_noise = c_noise.view(b, 1)
        
        Fout = self.net(c_in * x, c_noise)
        
        x0_hat = c_skip * x + c_out * Fout
        
        return x0_hat