import os
from EDM import get_edm_cifar_uncond_ve
import torch
import os
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.distributed import all_gather
from networks.resnet import ResNet18
from networks.WRN import WideResNet
import argparse
from transformers import AutoModelForImageClassification
from sampler.edm import EDMStochasticSampler

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=None)
parser.add_argument("--dataset", type=str, default='cifar10')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_test", type=int, default=64)
parser.add_argument("--pgd_step", type=int, default=20)
parser.add_argument("--eot_step", type=int, default=10)
parser.add_argument("--inference_step", type=int, default=20)
parser.add_argument("--model_type", type=str, default='t7')
parser.add_argument("--sigma_max", type=float, default=0.1)
parser.add_argument("--random_strength", type=float, default=0)
parser.add_argument("--s_min", type=float, default=0.01)
parser.add_argument("--s_max", type=float, default=0.46)
parser.add_argument("--s_noise", type=float, default=1.007)
parser.add_argument("--ckpt", type=str, default='../../edm_cifar_uncond_ve.pt')
parser.add_argument("--save_folder", type=str, default=None)
FLAGS = parser.parse_args()

  
local_rank = FLAGS.local_rank
init_process_group(backend="nccl")
torch.cuda.set_device(FLAGS.local_rank)
world_size = torch.distributed.get_world_size()
device = torch.device("cuda", local_rank)

Num_inference_steps = FLAGS.inference_step
sigma_max = FLAGS.sigma_max
batch_size = FLAGS.batch_size
num_test = FLAGS.num_test
pgd_step = FLAGS.pgd_step
eot_step = FLAGS.eot_step
ckpt_dir = FLAGS.ckpt
dir_temp = os.path.splitext(os.path.basename(ckpt_dir))[0]
logger.add(f"../../edm_result_attack/{FLAGS.save_folder}/{dir_temp}/edm_ve_linf_attack/{FLAGS.model_type}/log_test_edm_{dir_temp}_random_strength_{FLAGS.random_strength}_sigma_max_{sigma_max}_step_{Num_inference_steps}_bs_{batch_size}_num_test_{num_test}_pgd_{pgd_step}_eot_{eot_step}.log")


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
    states_att = torch.load('../../origin.t7', map_location='cpu')  # Temporary t7 setting
    network_clf = states_att['net'].to(device)
elif FLAGS.model_type == "kzcls":
    network_clf = WideResNet().to(device)
    network_clf.load_state_dict(torch.load('../../natural.pt', map_location='cpu')['state_dict'])
elif FLAGS.model_type == "r18":
    network_clf = ResNet18().to(device)
    network_clf.load_state_dict(torch.load('../../lastpoint.pth.tar', map_location='cpu'))
elif FLAGS.model_type == "cifar100":
    states_att = torch.load('../../wide-resnet-28x10-cifar100.t7', map_location='cpu')
    network_clf = states_att['net'].to(device)
elif FLAGS.model_type == "vit":
    network_clf = AutoModelForImageClassification.from_pretrained("../../vit-base-patch16-224-in21k-finetuned-cifar10").to(device)
network_clf.eval()

edm_model = get_edm_cifar_uncond_ve(pretrained=True, ckpt_dir=ckpt_dir).to(device)
edm_model.eval()
edm_model.requires_grad = False
edm = EDMStochasticSampler(unet = edm_model,device=device,
                 grad_checkpoint=True).to(device)


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
            images_1 = edm.purify(x = sample_image + delta,sigma_max=sigma_max,num_steps=Num_inference_steps, S_churn=FLAGS.random_strength, S_min=FLAGS.s_min, S_max=FLAGS.s_max, S_noise=FLAGS.s_noise)
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
        images_1 = edm.purify(x = adv_out,sigma_max=sigma_max,num_steps=Num_inference_steps, S_churn=FLAGS.random_strength, S_min=FLAGS.s_min, S_max=FLAGS.s_max, S_noise=FLAGS.s_noise)
    with torch.no_grad():
        if FLAGS.model_type == "vit":
            yy = network_clf(transform_cifar10(images_1.to(device))).logits
        else:
            yy = network_clf(transform_cifar10(images_1.to(device)))
        pure_cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("pure_cls_acc:", str(pure_cls_acc))
        logger.info("pure_cls_acc:" + str(pure_cls_acc) + str({local_rank}))
        pure_cls_acc_list.append(pure_cls_acc)
  
    del images_1, yy, adv_out, delta, d, eot, sample_image, y_val
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
