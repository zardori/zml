import os
import argparse
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# -------------------------
# Parse command line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Generate a video with CogVideoX")
parser.add_argument("--output_dir", type=str, default=".", help="Directory where the output video will be saved")
parser.add_argument("--prompt", type=str, default="A beautiful mountainside with a lot of red snow", help="Text prompt")
parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the video")
parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion steps")
parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, "cogvideox_5b_motorbike_output.mp4")

# -------------------------
# Load the model
# -------------------------
print("Loading CogVideoX model...")
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
print("Model loaded successfully!")

# -------------------------
# Generate video
# -------------------------
print(f"Generating video for prompt: {args.prompt}")
video = pipe(
    prompt=args.prompt,
    num_frames=args.num_frames,
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.num_inference_steps
).frames[0]

# -------------------------
# Save video
# -------------------------
export_to_video(video, output_path, fps=8)
print(f"Video saved as {output_path}")
