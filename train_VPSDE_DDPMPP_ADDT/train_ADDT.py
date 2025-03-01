import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import UNet2DModel,DiffusionPipeline,ImagePipelineOutput
from diffusers.schedulers import ScoreSdeVpScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from PIL import Image

from unet_sde_score_estimation import NCSNpp
from diffusers.configuration_utils import FrozenDict

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='cifar10',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=1, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=5, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args

class ScoreSdeVpPipeline_Warp(DiffusionPipeline):
    def __init__(self, model, scheduler):
        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        
    def __call__(self, num_inference_steps=1000, batch_size=1, generator=None,output_type =None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (batch_size, channels, img_size, img_size)

        model = self.model.to(device)
        x = torch.randn(*shape).to(device)
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            t = t * torch.ones(shape[0], device=device)
            scaled_t = t * (num_inference_steps - 1)

            with torch.no_grad():
                result = model(x, scaled_t)
            x, x_mean = self.scheduler.step_pred(result, x, t)

        x_mean = (x_mean + 1.0) / 2.0
        x_mean = x_mean.clip(0.0, 1.0)
        x_mean = x_mean.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            x_mean = self.numpy_to_pil(x_mean)
        return ImagePipelineOutput(images=x_mean)
    
class ScoreSdeVpScheduler_Warp(ScoreSdeVpScheduler):    
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
    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
        mean_coefficient = torch.exp(log_mean_coeff[:, None, None, None])
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean_coefficient, std, log_mean_coeff


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def main(args):
    if args.dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    transform_cifar10 = transforms.Compose(
        [transforms.Normalize(mean, std)]
    )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    if args.dataset_name == 'cifar10':
        states_att = torch.load('origin.t7', map_location='cpu')  # Temporary t7 setting
    elif args.dataset_name == 'cifar100':
        states_att = torch.load('wide-resnet-28x10-cifar100.t7', map_location='cpu')  # Temporary t7 setting
    network_clf = states_att['net']

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                #ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                os.makedirs(os.path.join(output_dir, "unet_ema"), exist_ok=True)
                torch.save(ema_model.state_dict(),os.path.join(output_dir, "unet_ema","ema_net.pth"))
            os.makedirs(os.path.join(output_dir, "unet"), exist_ok=True)
            torch.save(unet.state_dict(),os.path.join(output_dir, "unet", "net.pth"))
            # for i, model in enumerate(models):
            #     #model.save_pretrained(os.path.join(output_dir, "unet"))
            #     os.makedirs(os.path.join(output_dir, "unet"), exist_ok=True)
            #     torch.save(model.state_dict(),os.path.join(output_dir, "unet","net.pth"))
            #     # make sure to pop weight so that corresponding model is not saved again
            #     weights.pop()
                
        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), NCSNpp)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            model.load_state_dict(torch.load(model_id+'/diffusion_model.pt', map_location='cpu'),strict=False)
            # for i in range(len(models)):
            #     # pop models so that they are not loaded again
            #     model = models.pop()

            #     # load diffusers style into model
            #     load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
            #     model.register_to_config(**load_model.config)

            #     model.load_state_dict(load_model.state_dict())
            #     del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        #accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    model_id = args.model_config_name_or_path
    import json
    # 读取 JSON 文件
    with open(model_id+'/config.json', 'r') as file:
        data = json.load(file)
    data = FrozenDict(data)
    model = NCSNpp(data)
    model.load_state_dict(torch.load(model_id+'/diffusion_model.pt', map_location='cpu'),strict=False)


    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=NCSNpp,
            model_config=model.config,
        )


    noise_scheduler = ScoreSdeVpScheduler_Warp(num_train_timesteps=args.ddpm_num_steps)
    
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset_name == 'cifar10':
        def transform_images(examples):
            images = [augmentations(image.convert("RGB")) for image in examples["img"]]
            return {"input": images, "label": examples["label"]}
    elif args.dataset_name == 'cifar100':
        def transform_images(examples):
            images = [augmentations(image.convert("RGB")) for image in examples["img"]]
            return {"input": images, "label": examples["fine_label"]}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    network_clf.to(model.device)
    network_clf.eval()
    ppp = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(model.device)
    epsilon = (8 / 255.) / ppp
    alpha = (4 / 255.) / ppp
    upper_limit = ((1 - ppp)/ ppp)
    lower_limit = ((0 - ppp)/ ppp)
       
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    def set_dropout(model, p):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p

    if args.dropout>0:
        set_dropout(model, args.dropout)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"]
            clean_labels = batch['label']
            
            bsz = clean_images.shape[0]
            timesteps = torch.rand(bsz, device=clean_images.device) * (1 - noise_scheduler.config.sampling_eps) + noise_scheduler.config.sampling_eps
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            mean_coefficient, std, log_mean_coeff = noise_scheduler.marginal_prob(timesteps)
            # ratio = (mean_coefficient / std[:, None, None, None]).clip(0,10)
            advnoise_ratio = torch.clamp(0.03 * (mean_coefficient / std[:, None, None, None]), 0, 0.3)
            
            delta = torch.randn_like(clean_images)
            eot = 0
            for _ in range(5):
                for _ in range(1):
                    delta.requires_grad = True
                    noise = torch.randn(
                        clean_images.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
                    ).to(clean_images.device)
                    
                    mean_vp = mean_coefficient * (clean_images)
                    noisy_images = mean_vp + std[:, None, None, None] * (noise * (1 - advnoise_ratio ** 2) ** 0.5 + delta * advnoise_ratio)
                    scaled_t = timesteps * (args.ddpm_num_steps - 1)
                    with accelerator.autocast():
                        noise_pred = model(noisy_images, scaled_t)
                        original_samples = (noisy_images - std[:, None, None, None] * noise_pred) / mean_coefficient

                        network_clf_pred = network_clf(transform_cifar10(original_samples/2 + 0.5))
                        loss_cls = F.cross_entropy(network_clf_pred, clean_labels)
                        loss_l2 = F.mse_loss(noise_pred, noise)
                        balance = 1
                        loss = balance * loss_cls + (1 - balance) * loss_l2
                    accelerator.backward(loss, retain_graph=True)
                    grad = delta.grad.detach()
                    eot += grad
                    delta.grad.zero_()
                with torch.no_grad():
                    psedue_noise = torch.randn_like(delta)

                    psedue_noise = torch.sort(psedue_noise.reshape(psedue_noise.shape[0] * psedue_noise.shape[1], -1))[0]
                    psedue_order = torch.sort(eot.reshape(eot.shape[0] * eot.shape[1], -1))[1]
                    psedue_order = torch.sort(psedue_order)[1]

                    psedue_noise = torch.gather(psedue_noise, 1, psedue_order)
                    delta = psedue_noise.reshape(-1, 3, 32, 32).detach()


            with accelerator.accumulate(model):
                optimizer.zero_grad()
                noise = torch.randn(
                    clean_images.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
                ).to(clean_images.device)
                #mean_vp = mean_coefficient * (clean_images)
                noisy_images = mean_vp + std[:, None, None, None] * (noise * (1 - advnoise_ratio ** 2) ** 0.5 + delta * advnoise_ratio)
                scaled_t = timesteps * (args.ddpm_num_steps - 1)
                with accelerator.autocast():
                    noise_pred = model(noisy_images, scaled_t)
                    losses = torch.square(-noise_pred + noise * (1 - advnoise_ratio ** 2) ** 0.5 + delta * advnoise_ratio)
                    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
                    loss = torch.mean(losses)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = ScoreSdeVpPipeline_Warp(
                    model=unet,
                    scheduler=noise_scheduler,
                )

                generator = None#torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())
                    
                def make_grid(images, rows, cols):
                    w, h = images[0].size
                    grid = Image.new('RGB', size=(cols*w, rows*h))
                    for i, image in enumerate(images):
                        grid.paste(image, box=(i%cols*w, i//cols*h))
                    return grid
                
                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")
                
                # Make a grid out of the images
                image_grid = make_grid(pipeline.numpy_to_pil(images), rows=4, cols=4)
                # Save the images
                test_dir = os.path.join(args.output_dir, "samples")
                os.makedirs(test_dir, exist_ok=True)
                image_grid.save(f"{test_dir}/{epoch:04d}.png")
                
                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = ScoreSdeVpPipeline_Warp(
                    model=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)
                os.makedirs(os.path.join(args.output_dir, "unet"), exist_ok=True)
                torch.save(unet.state_dict(),os.path.join(args.output_dir, "unet", "net.pth"))

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
