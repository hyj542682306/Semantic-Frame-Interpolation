import torch
from src.models.model_manager import ModelManager
from src.models.utils import load_state_dict
from src.pipelines.wan_fi_custom import CustomWanFrameInterpolationPipeline
from src.data.video import save_video
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lora", required=True, type=str)
parser.add_argument("--prompt", required=True, type=str)
parser.add_argument("--negative_prompt", default="", type=str)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--frame", default=81, type=int)
parser.add_argument("--fps", default=30, type=int)
parser.add_argument("--height", default=None, type=int)
parser.add_argument("--width", default=None, type=int)
parser.add_argument("--st_img", required=True, type=str)
parser.add_argument("--ed_img", required=True, type=str)
parser.add_argument("--output", default="output.mp4", type=str)
args = parser.parse_args()

model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,
)
model_manager.load_models(
    [
        [
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)

pipe = CustomWanFrameInterpolationPipeline.from_model_manager(
    model_manager, 
    torch_dtype=torch.bfloat16, 
    device="cuda"
)

lora_state_dict = load_state_dict(args.lora)
missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(lora_state_dict, strict=False)

pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)


image_start = Image.open(args.st_img)
image_end = Image.open(args.ed_img)

if args.height is not None:
    assert args.width is not None
    height = args.height
    width = args.width
else:
    width, height = start_image.size

with torch.no_grad():
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image_start=image_start,
        input_image_end=image_end,
        num_frames=args.frame,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed, tiled=True
    )
save_video(video, args.output, fps=args.fps, quality=5)