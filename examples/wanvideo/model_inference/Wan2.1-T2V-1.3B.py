import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path='/home/jackie/Documents/UCLA/DiffSynth-Studio/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors'),
        ModelConfig(path='/home/jackie//Documents/UCLA/DiffSynth-Studio/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'),
        ModelConfig(path='/home/jackie//Documents/UCLA/DiffSynth-Studio/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'),
    ],
)
pipe.enable_vram_management()

# Text-to-video
video = pipe(
    prompt="A person walks past a brick wall with graffiti in red spray paint that says: ‘WE EXIST TOO",
    negative_prompt="extra limb, low quality, unreable texts",
    seed=0, tiled=True,
)
save_video(video, "before_.mp4", fps=15, quality=5)


