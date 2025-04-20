# ADDT/Fixed AutoAttack
Official code for ICLR2025 Paper "[Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective](https://openreview.net/forum?id=shqjOIK3SA)"


##

Comparison of attack trajectories under different evaluation settings.

![](./image/attack_illu.png "attack direction.")
Losslandscape

![](./image/main_img.png "Adversarial attack visualization.")

Adversarial Denoising Diffusion Training (ADDT)

![](./image/graph_loss.png "Adversarial Denoising Diffusion Training.")
RGBM

<img src="./image/gaussian_resample.png" width="300">

# Requirements
```
* transformers
* pytorch>=2.0.1
* numpy
* tqdm
* loguru
* diffusers>=0.20.0
* torchvision
* click>=8.0
* pillow>=8.3.1
* scipy>=1.7.1
```
  
# Directory Structure
```
running scripts are in the corresponding directory.
### All testing code(attack/clean evaluation for DDPM/DDIM/VPSDE/edm)
test/

├─ main_code: codes for attack/clean evaluation

├─ scripts_*: scripts for running the code, just "cd" to the directory and run the script, e.g. "cd test/script_attack && sh ddpm_run_attack_linf.sh"
### Code for calculating FID
fid/
### All training code for DDPM/DDIM/VPSDE/edm clean/ADDT
train_DDPM_DDIM/

train_VPSDE_DDPMPP/

train_edm_ADDT/
### Images for README.md
image/
```

## Citation
```
@inproceedings{
liu2025towards,
title={Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective},
author={Yiming Liu and Kezhao Liu and Yao Xiao and ZiYi Dong and Xiaogang Xu and Pengxu Wei and Liang Lin},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=shqjOIK3SA}
}
```


