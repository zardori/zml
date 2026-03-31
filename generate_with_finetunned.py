import os
import argparse
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from peft import PeftModel
import pandas as pd


# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate videos with the fine-tuned model on a given prompt set")
parser.add_argument("--output_dir", type=str, default=".", help="Directory where the output videos will be saved")
parser.add_argument("--prompt_dir", type=str, default="./vbench_prompts", help="Directory with prompts")
parser.add_argument("--seeded_prompt_file", type=str, default=None, help="Path to CSV file with 'prompt' and 'seed' columns")
parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the video")
parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion steps")
parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
parser.add_argument("--model_id", type=str, default="THUDM/CogVideoX-5b", help="Base model ID or path")
parser.add_argument("--model_checkpoint", type=str, default='./cogvideox_erasure_lora_nudity', help="Path to model checkpoint")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_prompts", type=int, default=None, help="Number of prompts to generate")
args = parser.parse_args()

# Ensure output directory exists and set torch seed
os.makedirs(args.output_dir, exist_ok=True)
torch.manual_seed(args.seed)

# Load the model
print("Loading CogVideoX model...")
pipe = CogVideoXPipeline.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
transformer = PeftModel.from_pretrained(pipe.transformer, args.model_checkpoint)
pipe.transformer = transformer
transformer.eval()
print("Model loaded successfully!")

# Generate videos
if args.seeded_prompt_file:
    # Load prompts and seeds from CSV file
    df = pd.read_csv(args.seeded_prompt_file)
    if args.n_prompts:
        df = df.head(args.n_prompts)
    for idx, row in df.iterrows():
        prompt = row['prompt']
        seed = int(row['seed'])

        print(f"Generating video for prompt: {prompt} with seed: {seed}")
        print(args.num_frames, args.guidance_scale, args.num_inference_steps)

        video = pipe(
            prompt=prompt,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        ).frames[0]

        cur_save_path = f'{args.output_dir}/{idx}-{seed}.mp4'

        export_to_video(video, cur_save_path, fps=args.fps)
        print(f"Video saved as {cur_save_path}")
else:
    dimension_list = ['our_prompt_test_set'] #'object_class' #'subject_consistency
    for dimension in dimension_list: 
        
        with open(f'{args.prompt_dir}/{dimension}.txt', 'r') as f:
            prompt_list = f.readlines()
        prompt_list = [prompt.strip() for prompt in prompt_list]
        
        for j, prompt in enumerate(prompt_list):
            for index in range(5):
                print(f"Generating video for prompt: {prompt}")
                print(args.num_frames, args.guidance_scale, args.num_inference_steps)
                seed = args.seed + index
                video = pipe(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator(device="cuda").manual_seed(seed)
                ).frames[0]

                cur_save_path = f'{args.output_dir}/test---{j}-{index}.mp4'
                
                export_to_video(video, cur_save_path, fps=args.fps)
                print(f"Video saved as {cur_save_path}")