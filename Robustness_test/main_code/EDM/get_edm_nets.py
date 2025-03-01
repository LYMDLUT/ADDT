import torch
from .edm_nets import EDMPrecond


def get_edm_cifar_cond(pretrained=True):
    network_kwargs = dict(
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
    )
    model = EDMPrecond(img_resolution=32, img_channels=3, label_dim=10, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./resources/checkpoints/EDM/edm_cifar_cond.pt"))
    return model


def get_edm_cifar_uncond(pretrained=True, ckpt_dir=None):
    network_kwargs = dict(
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
    )
    model = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load(ckpt_dir, map_location='cpu'))
    return model


def get_edm_cifar_uncond_ve(pretrained=True, ckpt_dir=None):
    network_kwargs = dict(
        model_type="SongUNet",
        embedding_type="fourier",
        encoder_type="residual",
        decoder_type="standard",
        channel_mult_noise=2,
        resample_filter=[1, 3, 3, 1],
        model_channels=128,
        channel_mult=[2, 2, 2],
        augment_dim=9,
    )
    model = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load(ckpt_dir, map_location='cpu'))
    return model


def get_edm_imagenet_64x64_cond(pretrained=True):
    network_kwargs = dict(model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4])
    model = EDMPrecond(img_resolution=64, img_channels=3, label_dim=1000, **network_kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./resources/checkpoints/EDM/edm-imagenet-64x64-cond.pt"))
    return model
