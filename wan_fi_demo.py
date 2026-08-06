import torch
from src.models.model_manager import ModelManager
from src.models.utils import load_state_dict
from src.pipelines.wan_fi_custom import CustomWanFrameInterpolationPipeline
from src.data.video import save_video
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lora_path", default="ckpts", type=str)
parser.add_argument("--model_path", default="models/Wan2.1-I2V-14B-480P", type=str)
parser.add_argument("--prompt", default="", type=str)
parser.add_argument("--negative_prompt", default="", type=str)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--frame", default=81, type=int)
parser.add_argument("--fps", default=30, type=int)
parser.add_argument("--height", default=None, type=int)
parser.add_argument("--width", default=None, type=int)
parser.add_argument("--st_img", required=True, type=str)
parser.add_argument("--ed_img", required=True, type=str)
parser.add_argument("--output", default="output.mp4", type=str)
args = parser.parse_args()

loras = [
    f"{args.lora_path}/uni.ckpt",
    f"{args.lora_path}/exp_0.ckpt",
    f"{args.lora_path}/exp_1.ckpt",
    f"{args.lora_path}/exp_2.ckpt",
    f"{args.lora_path}/exp_3.ckpt",
    f"{args.lora_path}/exp_4.ckpt",
    f"{args.lora_path}/exp_5.ckpt"
]
target_modules = ['q', 'k', 'v', 'o', 'ffn_0', 'ffn_2']

model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [f"{args.model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,
)
model_manager.load_models(
    [
        [
            f"{args.model_path}/diffusion_pytorch_model-00001-of-00007.safetensors",
            f"{args.model_path}/diffusion_pytorch_model-00002-of-00007.safetensors",
            f"{args.model_path}/diffusion_pytorch_model-00003-of-00007.safetensors",
            f"{args.model_path}/diffusion_pytorch_model-00004-of-00007.safetensors",
            f"{args.model_path}/diffusion_pytorch_model-00005-of-00007.safetensors",
            f"{args.model_path}/diffusion_pytorch_model-00006-of-00007.safetensors",
            f"{args.model_path}/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{args.model_path}/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)

pipe = CustomWanFrameInterpolationPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)

# load lora
target_lora_state_dict = {}
for idx, lora_path in enumerate(loras):
    lora_state_dict = load_state_dict(lora_path)
    for key, value in lora_state_dict.items():
        if idx == 0:
            for tar_key in target_modules:
                if "uni_lora_" + tar_key in key:
                    target_lora_state_dict.update(
                        {key.replace("uni_lora_" + tar_key, "lora_" + tar_key + ".0"): value}
                    )
        else:
            for tar_key in target_modules:
                if "exp_lora_" + tar_key in key:
                    target_lora_state_dict.update(
                        {key.replace("exp_lora_" + tar_key, "lora_" + tar_key + "." + str(idx)): value}
                    )

pipe.denoising_model().load_state_dict(target_lora_state_dict, strict=False)
pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)


image_start = Image.open(args.st_img)
image_end = Image.open(args.ed_img)

if args.height is not None:
    assert args.width is not None
    height = args.height
    width = args.width
else:
    width, height = image_start.size

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
save_video(video, args.output, fps=args.fps, quality=9)