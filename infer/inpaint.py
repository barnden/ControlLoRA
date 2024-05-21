import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, DDIMScheduler
from models.control_lora import ControlLoRAModel
from train.inpaint import config
from diffusers.utils import load_image
import torch.utils.data as data
from co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation
from pathlib import Path
from typing import List
from posewarp import Warper
from diffusers.models.controlnet import ControlNetModel


seed = 241

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# dtype = torch.float32
# image = Image.open("./docs/imgs/face_landmarks1.jpeg")

base_model = "runwayml/stable-diffusion-v1-5"

unet = UNet2DConditionModel.from_pretrained(
    base_model, subfolder="unet", torch_dtype=dtype
)

pretrained_model_name_or_path = config.output_dir
if len(glob.glob(f'./{config.output_dir}/*.safetensors')) == 0:
    # Get the most recent checkpoint
    dirs = os.listdir(config.output_dir) if os.path.exists(config.output_dir) else []
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))

    if len(dirs) > 0:
        pretrained_model_name_or_path = os.path.join(config.output_dir, dirs[-1], 'control-lora')
    else:
        raise "Model not found"


control_lora = ControlLoRAModel.from_pretrained(
    pretrained_model_name_or_path, torch_dtype=dtype
)

if config.pretrained_controlnet_name_or_path:
    control_unet = ControlNetModel.from_pretrained(
        config.pretrained_controlnet_name_or_path, cache_dir=config.cache_dir, torch_dtype=dtype
    )
    control_lora.tie_weights(control_unet)
else:
    control_lora.tie_weights(unet)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model, unet=unet, controlnet=control_lora, safety_checker=None, torch_dtype=dtype
).to(device)

control_lora.bind_vae(pipe.vae)

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

from custom_datasets.CO3D import WarpCO3DDataset

dataset = WarpCO3DDataset(offset=5)

data = dataset[2812]

generator = torch.Generator(pipe.device).manual_seed(seed)

image = pipe(
    prompt=data["prompt"],
    image=data['pil_condition'],
    generator=generator,
    num_inference_steps=20).images[0]

data['pil_original'].save('__original.png')
data['pil_warp'].save("__warped.png")
data['pil_condition'].save('__condition.png')
data['pil_target'].save('__target.png')
image.save('__inpaint_tile.png')
