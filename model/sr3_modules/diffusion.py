import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import kornia

# DDNM
def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return th.stack((x*coef/base, x*coef/base, x*coef/base), 1)    
    
# def PatchUpsample(x, scale):
#     n, c, h, w = x.shape
#     x = torch.zeros(n,c,h,scale,w,scale) + x.view(n,c,h,1,w,1)
#     return x.view(n,c,scale*h,scale*w)

def PatchUpsample(x, scale):
    n, c, h, w = x.size()
    x = torch.zeros(n,c,h,scale,w,scale, device=x.device) + x.view(n,c,h,1,w,1)
    return x.permute(0,1,2,4,3,5).contiguous().view(n,c,h*scale,w*scale)

# Implementation of A and its pseudo-inverse Ap    
IR_mode = "super resolution"
scale = 8
if IR_mode=="colorization":
    A = color2gray
    Ap = gray2color
    
elif IR_mode=="inpainting":
    A = lambda z: z*mask
    Ap = A
      
elif IR_mode=="super resolution":
    A = torch.nn.AdaptiveAvgPool2d((128//scale,128//scale))
    Ap = lambda z: PatchUpsample(z, scale)

elif IR_mode=="old photo restoration":
    A1 = lambda z: z*mask
    A1p = A1
    
    A2 = color2gray
    A2p = gray2color
    
    A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    A3p = lambda z: PatchUpsample(z, scale)
    
    A = lambda z: A3(A2(A1(z)))
    Ap = lambda z: A1p(A2p(A3p(z)))
# Core Implementation of DDNM+, simplified denoising solution (Section 3.3).
# For more accurate denoising, please refer to the paper (Appendix I) and the source code.

def ddnm_plus_core(x0t, y, sigma_t, a_t, sigma_y=0):
    # Eq 19
    if sigma_t >= a_t*sigma_y: 
        lambda_t = 1
        gamma_t = sigma_t**2 - (a_t*lambda_t*sigma_y)**2
    else:
        lambda_t = sigma_t/(a_t*sigma_y)
        gamma_t = 0
        
    # Eq 17    
    x0t = x0t + lambda_t*Ap(y - A(x0t))
    
    return x0t, gamma_t
#---------------------------------------------------------------------------#

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        self.ssim_loss =kornia.losses.SSIM(11, reduction='mean')
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        '''
        设置新的噪声张量
        args:
            schedule_opt: 噪声张量的参数
            device: 设备
        '''
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        '''
        从噪声预测开始图像
        args:
            x_t: 输入图像
            t: 时间步
            noise: 噪声
        '''
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        '''
        计算 q(x_{t-1} | x_t, x_0) 的均值和方差
        args:
            x_start: 输入图像
            x_t: 输入图像
            t: 时间步
        '''
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        '''
        计算 p(x_t | x_{t-1}) 的均值和方差
        args:
            x: 输入图像
            t: 时间步
            clip_denoised: 是否对生成图像进行截断
            condition_x: 条件图像 
        '''
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        '''
        生成样本
        args:
            x: 输入图像
            t: 时间步
            clip_denoised: 是否对生成图像进行截断
            condition_x: 条件图像
        '''
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x) # 生成一个与输入 x 形状相同的噪声张量
        return model_mean + noise * (0.5 * model_log_variance).exp() #  这行代码是生成样本

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        '''
        生成样本
        args:
            x_in: 输入图像
            continous: 是否输出去噪过程
        '''
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in['1_SR']
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in['1_SR']
            shape = x.shape
            img = torch.randn(shape, device=device) # 噪声
            ret_img = x

            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x_in['1_2_3_SR'])
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img # 输出去噪过程
        else:
            return ret_img[-1] # 去噪后的图像

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        """
        计算损失函数

        参数：
        x_in (dict): 输入的图像数据，包含'HR'键，值为高分辨率图像张量
        noise (torch.Tensor, optional): 噪声张量，默认为None

        返回：
        loss (torch.Tensor): 损失张量
        """

        x_start = x_in['HR']  # 获取高分辨率图像
        [b, c, h, w] = x_start.shape  # 获取图像张量的形状
        t = np.random.randint(1, self.num_timesteps + 1)  # 随机选择时间步数
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)  # 生成连续的平方根累积乘积张量
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)  # 调整平方根累积乘积张量的形状

        noise = default(noise, lambda: torch.randn_like(x_start))  # 随机生成噪声张量，若未提供噪声则默认为高斯噪声
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)  # 通过 q_sample 方法添加噪声到高分辨率图像上

        if not self.conditional:  # 如果不是条件模式
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)  # 通过 denoise_fn 方法对添加噪声的图像进行去噪
        else:  # 如果是条件模式
            # x_recon = self.denoise_fn(
            #     torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)  # 通过 denoise_fn 方法将 Super-Resolution (SR) 图像和添加噪声的图像进行拼接再进行去噪

            x_recon = self.denoise_fn(
                torch.cat([x_in['1_2_3_SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod) # 通过 denoise_fn 方法将红外 (IR) 和可见光 (VI) 的 SR 图像和添加噪声的图像进行拼接再进行去噪
        loss = self.loss_func(noise, x_recon) + self.ssim_loss(x_start, x_recon) # 计算损失值
        return loss  # 返回损失值

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
