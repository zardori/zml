import os

file_path = "prompts/vbench_prompts/cogvideox_nudity.txt"
with open(file_path, 'r') as f:
    prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]

file_path = "prompts/vbench_prompts/working_prompts.txt"
with open(file_path, 'r') as f:
    worker_prompt_list = f.readlines()
    worker_prompt_list = [idx.strip() for idx in worker_prompt_list]
    worker_prompt_list = set(map(int, worker_prompt_list))

new_prompts = []
for idx, prompt in enumerate(prompt_list):
    if idx in worker_prompt_list:
        new_prompts.append(prompt)

# Save the new prompts to a new file
output_file_path = "prompts/vbench_prompts/filtered_cogvideox_nudity.txt"
if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))

with open(output_file_path, 'w') as f:
    for prompt in new_prompts:
        f.write(prompt + '\n')

# The above code reads two files: one containing the original prompts and another containing the indices of the prompts that workers found to be inappropriate. It then filters the original prompts based on the indices provided by the workers and saves the filtered prompts to a new file.