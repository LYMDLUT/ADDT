import os
from diffusers import DiffusionPipeline
from diffusers.schedulers import ScoreSdeVpScheduler
from diffusers.configuration_utils import FrozenDict
from unet_sde_score_estimation import NCSNpp
from diffusers.utils.torch_utils import randn_tensor
import torch
import math
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.distributed import all_gather
from networks.resnet import ResNet18
from networks.WRN import WideResNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_test", type=int, default=64)
parser.add_argument("--pgd_step", type=int, default=20)
parser.add_argument("--eot_step", type=int, default=10)
parser.add_argument("--denoise_strength", type=float, default=0.1)
parser.add_argument("--inference_step", type=int, default=1000)
parser.add_argument("--model_id", type=str, default='../../cifar10-ddpmpp-deep-vp')
parser.add_argument("--model_type", type=str, default='t7')
parser.add_argument("--fix_type", type=str, default='large')
FLAGS = parser.parse_args()

local_rank = FLAGS.local_rank
init_process_group(backend="nccl")
torch.cuda.set_device(FLAGS.local_rank)
world_size = torch.distributed.get_world_size()
device = torch.device("cuda", local_rank)


Denoise_strength = FLAGS.denoise_strength
Num_inference_steps = FLAGS.inference_step
batch_size = FLAGS.batch_size
num_test = FLAGS.num_test
pgd_step = FLAGS.pgd_step
eot_step = FLAGS.eot_step

logger.add(f"../vp_lock/attack_result/{FLAGS.model_type}/log_test_fix_{FLAGS.fix_type}_ddpmpp_vp_t_{Denoise_strength}_step_{Num_inference_steps}_|{FLAGS.model_id.split('/')[-1]}|_bs_{batch_size}_num_test_{num_test}_pgd_{pgd_step}_eot_{eot_step}.log")


model_config = "../../cifar10-ddpmpp-deep-vp"
model_id = FLAGS.model_id
class ScoreSdeVpPipeline_Warp(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def __call__(self,
                 sample_image,
                 batch_size: int = 1,
                 timesteps_list = None,
                 reverse_noise = None,
                 generator=None):

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (batch_size, channels, img_size, img_size)

        for t, lock_noise in zip(timesteps_list,reverse_noise):
            t = t * torch.ones(shape[0], device=device)
            scaled_t = t * (Num_inference_steps - 1)
            
            result = checkpoint(self.model, sample_image, scaled_t)
            sample_image, x_mean = self.scheduler.step_pred(result, sample_image, t, lock_noise=lock_noise)

        x_mean = (x_mean + 1.0) / 2.0
        return x_mean
    
class ScoreSdeVpScheduler_Warp(ScoreSdeVpScheduler):    
    def set_alpha(self):
        self.discrete_betas = torch.linspace(self.config.beta_min / Num_inference_steps, self.config.beta_max / Num_inference_steps, Num_inference_steps).to(device)
        self.alphas = 1. - self.discrete_betas
        #self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        timesteps = timesteps.to(original_samples.device)
        log_mean_coeff = -0.25 * timesteps ** 2 * (self.config.beta_max - self.config.beta_min) - 0.5 * timesteps * self.config.beta_min
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * original_samples
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        noisy_samples = mean + std[:, None, None, None] * noise
        return noisy_samples
    def step_pred(self, score, x, t, lock_noise=None,generator=None):
        log_mean_coeff = (
            -0.25 * t**2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
        )
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        std = std.flatten()
        while len(std.shape) < len(score.shape):
            std = std.unsqueeze(-1)
        score = -score / std

        # compute
        dt = -1.0 / len(self.timesteps)

        beta_t = self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)
        beta_t = beta_t.flatten()
        while len(beta_t.shape) < len(x.shape):
            beta_t = beta_t.unsqueeze(-1)
        drift = -0.5 * beta_t * x

        diffusion = torch.sqrt(beta_t)
        drift = drift - diffusion**2 * score
        x_mean = x + drift * dt

        # add noise
        #noise = randn_tensor(x.shape, layout=x.layout, generator=generator, device=x.device, dtype=x.dtype)
        noise = lock_noise.to(x.device)
        x = x_mean + diffusion * math.sqrt(-dt) * noise
        return x, x_mean


import json
# 读取 JSON 文件
with open(model_config+'/config.json', 'r') as file:
    data = json.load(file)
data = FrozenDict(data)
unet = NCSNpp(data).to(device)
mm = torch.load(model_id + '/unet/net.pth', map_location='cpu')
new_state_dict = {}
for key, value in mm.items():
    if key.startswith("module."):
        new_key = key[len("module."):]  # 删除前缀
    else:
        new_key = key
    new_state_dict[new_key] = value
unet.load_state_dict(new_state_dict,strict=False)


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
if FLAGS.model_type == "t7":
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
elif FLAGS.model_type == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
elif FLAGS.model_type == "r18":
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
 
mu = torch.tensor(mean).view(3, 1, 1).to(device)
std1 = torch.tensor(std).view(3, 1, 1).to(device)
ppp = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
upper_limit = ((1 - ppp)/ ppp)
lower_limit = ((0 - ppp)/ ppp)
# upper_limit = ((1 - mu)/ std1)
# lower_limit = ((0 - mu)/ std1)
transform_cifar10 = transforms.Compose(
    [transforms.Normalize(mean, std)]
)
cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
cifar10_test.data = cifar10_test.data[:num_test]
cifar10_test.targets = cifar10_test.targets[:num_test]

sampler = DistributedSampler(cifar10_test, num_replicas=world_size, rank=local_rank)
cifar10_test_loader = DataLoader(
    cifar10_test, shuffle=False, num_workers=5, batch_size=batch_size, sampler=sampler, drop_last=True)

if FLAGS.model_type == "t7":
    states_att = torch.load('../../origin.t7', map_location='cpu')  # Temporary t7 setting
    network_clf = states_att['net'].to(device)
elif FLAGS.model_type == "kzcls":
    network_clf = WideResNet().to(device)
    network_clf.load_state_dict(torch.load('../../natural.pt', map_location='cpu')['state_dict'])
else:
    network_clf = ResNet18().to(device)
    network_clf.load_state_dict(torch.load('../../lastpoint.pth.tar', map_location='cpu'))
network_clf.eval()

noise_scheduler = ScoreSdeVpScheduler_Warp(num_train_timesteps=1000)
noise_scheduler.set_alpha()
noise_scheduler.set_timesteps(num_inference_steps=Num_inference_steps) 
timesteps_list = torch.FloatTensor(noise_scheduler.timesteps[(round((1-Denoise_strength)*len(noise_scheduler.timesteps))-1):])
timesteps = timesteps_list[0]

ncsn_vp = ScoreSdeVpPipeline_Warp(unet, noise_scheduler).to(device)


epsilon = (8 / 255.) / ppp
alpha = (2 / 255.) / ppp
def clamp1(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

cls_acc_list = []
pure_cls_acc_list = []
cls_init_acc_list = []

for sample_image, y_val in tqdm(cifar10_test_loader, colour='yellow'):
    sample_image = sample_image.to(device)
    y_val = y_val.to(device)
    noise_global = torch.randn(sample_image.shape).to(device)
    reverse_noise = torch.randn([1000, batch_size, *sample_image.shape[1:]])
    with torch.no_grad():
        yy_init= network_clf(transform_cifar10(sample_image/2+0.5))
        cls_acc_init = sum((yy_init.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("cls_init_acc:", str(cls_acc_init.cpu().item()))
        logger.info("cls_init_acc:"+str(cls_acc_init.cpu().item())+str({local_rank}))
        cls_init_acc_list.append(cls_acc_init.cpu().item())
        
    delta = torch.zeros_like(sample_image)
    delta.requires_grad = True
    for _ in tqdm(range(pgd_step), colour='red'):
        eot = 0
        for _ in range(1):
            noise = torch.randn(sample_image.shape).to(device)
            noisy_image = noise_scheduler.add_noise(sample_image + delta, noise_global, timesteps.repeat(sample_image.shape[0]))
            images_1= ncsn_vp(sample_image=noisy_image.to(device), batch_size=noisy_image.shape[0], timesteps_list=timesteps_list,reverse_noise=reverse_noise)

            tmp_in = transform_cifar10(images_1)
            tmp_out = network_clf(tmp_in)
            loss = F.cross_entropy(tmp_out, y_val)
            loss.backward()
            grad = delta.grad.detach()
            eot += grad
            delta.grad.zero_()
            ncsn_vp.model.zero_grad()
            network_clf.zero_grad()
        d = clamp1(delta + alpha * eot.sign(), -epsilon, epsilon)
        d = clamp1(d, lower_limit - sample_image, upper_limit - sample_image)
        delta.data = d

    adv_out = (sample_image + delta)
    
    with torch.no_grad():
        yy = network_clf(transform_cifar10(adv_out/2+0.5))
        cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("cls_acc:", str(cls_acc.cpu().item()))
        logger.info("cls_acc:"+str(cls_acc.cpu().item())+str({local_rank}))
        cls_acc_list.append(cls_acc.cpu().item())
    
    with torch.no_grad():
        noise = torch.randn(sample_image.shape).to(device)
        noisy_image = noise_scheduler.add_noise(adv_out, noise_global, timesteps.repeat(sample_image.shape[0]))
        images_1 = ncsn_vp(sample_image=noisy_image.to(device), batch_size=noisy_image.shape[0], timesteps_list=timesteps_list,reverse_noise=reverse_noise)
             
    with torch.no_grad():
        yy = network_clf(transform_cifar10(images_1.to(device)))
        pure_cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("pure_cls_acc:", str(pure_cls_acc))
        logger.info("pure_cls_acc:" + str(pure_cls_acc) + str({local_rank}))
        pure_cls_acc_list.append(pure_cls_acc)
  
    del images_1, yy, noisy_image, adv_out, delta, d, eot, noise, sample_image, y_val
    torch.distributed.barrier()
    torch.cuda.empty_cache()

print("=====================================")
cls_init_acc_list = torch.tensor(cls_init_acc_list).to(device)
gathered_cls_init_acc_list = [cls_init_acc_list.clone() for _ in range(world_size)]
all_gather(gathered_cls_init_acc_list, cls_init_acc_list)
cls_init_acc_list = torch.cat(gathered_cls_init_acc_list, dim=0)

cls_acc_list = torch.tensor(cls_acc_list).to(device)
gathered_cls_acc_list = [cls_acc_list.clone() for _ in range(world_size)]
all_gather(gathered_cls_acc_list, cls_acc_list)
cls_acc_list = torch.cat(gathered_cls_acc_list, dim=0)

pure_cls_acc_list = torch.tensor(pure_cls_acc_list).to(device)
gathered_pure_cls_acc_list = [pure_cls_acc_list.clone() for _ in range(world_size)]
all_gather(gathered_pure_cls_acc_list, pure_cls_acc_list)
pure_cls_acc_list = torch.cat(gathered_pure_cls_acc_list, dim=0)
if local_rank == 0:
    print("all_cls_init_acc:", "{:0.4f}".format(sum(cls_init_acc_list)/len(cls_init_acc_list)))
    logger.info("all_cls_init_acc"+"{:0.4f}".format(sum(cls_init_acc_list)/len(cls_init_acc_list)))
    print("all_cls_acc:", "{:0.4f}".format(sum(cls_acc_list)/len(cls_acc_list)))
    logger.info("all_cls_acc"+"{:0.4f}".format(sum(cls_acc_list)/len(cls_acc_list)))
    print("all_pure_cls_acc:", "{:0.4f}".format(sum(pure_cls_acc_list)/len(pure_cls_acc_list)))
    logger.info("all_pure_cls_acc"+"{:0.4f}".format(sum(pure_cls_acc_list)/len(pure_cls_acc_list)))
