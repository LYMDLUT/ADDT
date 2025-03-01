import os
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from tqdm import tqdm
from loguru import logger
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.distributed import all_gather
from networks.resnet import ResNet18
from networks.WRN import WideResNet
import argparse
from transformers import AutoModelForImageClassification

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--dataset", type=str, default='cifar10')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_test", type=int, default=64)
parser.add_argument("--pgd_step", type=int, default=20)
parser.add_argument("--eot_step", type=int, default=10)
parser.add_argument("--denoise_strength", type=float, default=0.1)
parser.add_argument("--inference_step", type=int, default=1000)
parser.add_argument("--model_id", type=str, default='../ckpt/ddpm_ema_cifar10')
parser.add_argument("--model_type", type=str, default='t7')
parser.add_argument("--fix_type", type=str, default='large')
parser.add_argument("--save_folder", type=str, default=None)
FLAGS = parser.parse_args()

local_rank = FLAGS.local_rank
init_process_group(backend="nccl")
torch.cuda.set_device(FLAGS.local_rank)
world_size = torch.distributed.get_world_size()
device = torch.device("cuda", local_rank)


Denoise_strength = FLAGS.denoise_strength
Num_inference_steps = FLAGS.inference_step

model_id = FLAGS.model_id

batch_size = FLAGS.batch_size
num_test = FLAGS.num_test
pgd_step = FLAGS.pgd_step
eot_step = FLAGS.eot_step

logger.add(f"../../result_attack/{FLAGS.save_folder}/ddpm_linf_attack/{FLAGS.model_type}/log_test_fix_{FLAGS.fix_type}_ddpm_t_{Denoise_strength}_step_{Num_inference_steps}_|{model_id.split('/')[-1]}|_bs_{batch_size}_num_test_{num_test}_pgd_{pgd_step}_eot_{eot_step}.log")


class DDPMPipeline_Img2Img(DDPMPipeline):
    def __call__(
        self,
        sample_image,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        timesteps_list = None,
        output_type: Optional[str] = "pil",
    ) -> Union[ImagePipelineOutput, Tuple]:

        image = sample_image
        # set step values
        self.unet.eval()
        for t in timesteps_list:
            # 1. predict noise model_output
            model_output = checkpoint(self.unet, image, t, None, False)[0]
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image1 = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        if output_type == "pil":
            image1 = self.numpy_to_pil(image1)
        return image, image1



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
elif FLAGS.model_type == "r18" or FLAGS.model_type == "vit":
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
 
mu = torch.tensor(mean).view(3, 1, 1).to(device)
std1 = torch.tensor(std).view(3, 1, 1).to(device)
ppp = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
upper_limit = ((1 - ppp)/ ppp)
lower_limit = ((0 - ppp)/ ppp)
# upper_limit = ((1 - mu)/ std1)
# lower_limit = ((0 - mu)/ std1)
if FLAGS.model_type == "vit":
    transform_cifar10 = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    )
else:
    transform_cifar10 = transforms.Compose(
        [transforms.Normalize(mean, std)]
    )
if FLAGS.dataset=='cifar10':
    cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif FLAGS.dataset=='cifar100':
    cifar10_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
cifar10_test.data = cifar10_test.data[:num_test]
cifar10_test.targets = cifar10_test.targets[:num_test]

sampler = DistributedSampler(cifar10_test, num_replicas=world_size, rank=local_rank)
cifar10_test_loader = DataLoader(
    cifar10_test, shuffle=False, num_workers=5, batch_size=batch_size, sampler=sampler, drop_last=True)

if FLAGS.model_type == "t7":
    states_att = torch.load('../origin.t7', map_location='cpu')  # Temporary t7 setting
    network_clf = states_att['net'].to(device)
elif FLAGS.model_type == "kzcls":
    network_clf = WideResNet().to(device)
    network_clf.load_state_dict(torch.load('../natural.pt', map_location='cpu')['state_dict'])
elif FLAGS.model_type == "r18":
    network_clf = ResNet18().to(device)
    network_clf.load_state_dict(torch.load('../lastpoint.pth.tar', map_location='cpu'))
elif FLAGS.model_type == "cifar100":
    states_att = torch.load('../wide-resnet-28x10-cifar100.t7', map_location='cpu')
    network_clf = states_att['net'].to(device)
elif FLAGS.model_type == "vit":
    network_clf = AutoModelForImageClassification.from_pretrained("../../vit-base-patch16-224-in21k-finetuned-cifar10").to(device)

network_clf.eval()

if FLAGS.fix_type == 'small':
    noise_schdeuler1 = DDPMScheduler(num_train_timesteps=1000)
else:
    noise_schdeuler1 = DDPMScheduler(num_train_timesteps=1000, variance_type="fixed_large")
    
noise_schdeuler1.set_timesteps(num_inference_steps=Num_inference_steps) 
timesteps_list = torch.LongTensor(noise_schdeuler1.timesteps[(round((1-Denoise_strength)*len(noise_schdeuler1.timesteps))-1):])
timesteps = timesteps_list[0]

unet = UNet2DModel.from_pretrained(model_id, subfolder='unet').to(device)
ddpm = DDPMPipeline_Img2Img(unet, noise_schdeuler1).to(device)


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
    with torch.no_grad():
        if FLAGS.model_type == "vit":
            yy_init= network_clf(transform_cifar10(sample_image/2+0.5)).logits
        else:
            yy_init= network_clf(transform_cifar10(sample_image/2+0.5))
        cls_acc_init = sum((yy_init.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("cls_init_acc:", str(cls_acc_init.cpu().item()))
        logger.info("cls_init_acc:"+str(cls_acc_init.cpu().item())+str({local_rank}))
        cls_init_acc_list.append(cls_acc_init.cpu().item())
        
    delta = torch.zeros_like(sample_image)
    delta.requires_grad = True
    for _ in tqdm(range(pgd_step), colour='red'):
        eot = 0
        for _ in range(eot_step):
            noise = torch.randn(sample_image.shape).to(device)
            noisy_image = noise_schdeuler1.add_noise(sample_image + delta, noise, timesteps)
            images_1, images_2 = ddpm(sample_image=noisy_image.to(device), batch_size=noisy_image.shape[0], timesteps_list=timesteps_list)

            tmp_in = transform_cifar10(images_1)
            if FLAGS.model_type == "vit":
                tmp_out = network_clf(tmp_in).logits
            else:
                tmp_out = network_clf(tmp_in)
            loss = F.cross_entropy(tmp_out, y_val)
            loss.backward()
            grad = delta.grad.detach()
            eot += grad
            delta.grad.zero_()
            ddpm.unet.zero_grad()
            network_clf.zero_grad()
        d = clamp1(delta + alpha * eot.sign(), -epsilon, epsilon)
        d = clamp1(d, lower_limit - sample_image, upper_limit - sample_image)
        delta.data = d

    adv_out = (sample_image + delta)
    
    with torch.no_grad():
        if FLAGS.model_type == "vit":
            yy = network_clf(transform_cifar10(adv_out/2+0.5)).logits
        else:
            yy = network_clf(transform_cifar10(adv_out/2+0.5))
        cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("cls_acc:", str(cls_acc.cpu().item()))
        logger.info("cls_acc:"+str(cls_acc.cpu().item())+str({local_rank}))
        cls_acc_list.append(cls_acc.cpu().item())
    
    with torch.no_grad():
        noise = torch.randn(sample_image.shape).to(device)
        noisy_image = noise_schdeuler1.add_noise(adv_out, noise, timesteps)
        images_1, images_2 = ddpm(sample_image=noisy_image.to(device), batch_size=noisy_image.shape[0], timesteps_list=timesteps_list)
             
    with torch.no_grad():
        if FLAGS.model_type == "vit":
            yy = network_clf(transform_cifar10(images_1.to(device))).logits
        else:
            yy = network_clf(transform_cifar10(images_1.to(device)))
        pure_cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("pure_cls_acc:", str(pure_cls_acc))
        logger.info("pure_cls_acc:" + str(pure_cls_acc) + str({local_rank}))
        pure_cls_acc_list.append(pure_cls_acc)
  
    del images_1, images_2, yy, noisy_image, adv_out, delta, d, eot, noise, sample_image, y_val
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
