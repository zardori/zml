# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The goal of this research project is to propose a method for effective concept unlearning from text to video models. The project uses CogVideoX-5b, a video diffusion transformer, as the primary model for experiments. Previously we tried to erase the "nudity" concept, now we focus on the "fire" concept. The real challenge is to erase the target concept without harming the model's performance. The project uses python 3.12 and uv for python packages. Experiments are run on PLGrid HPC infrastructure athena cluster via SLURM.

## Desired Repository Structure
```
zml/
├── zml/                         # shared "library" code
│   ├── unlearn/                 # scripts for unlearning
│   ├── precompute/              # scripts for precomputing latents used in unlearning
│   └── eval/                    # scripts and utils for evaluation
├── experiments/                 # one folder per experiment run
│   ├── exp001_esd_nudity/       
│   │   ├── config.yaml          # hyperparameters, dataset info, etc.
│   │   ├── outputs_{TIMESTAMP}  # generated videos, evaluation results, etc.
│   │   └── notes.md             # what was tried, what happened
│   ├── exp002_esd_fire_lora8/   
│   └── ...                      
├── scripts/                     # thin generic entrypoints (call zml/)
│   ├── unlearn.py               
│   ├── precompute.py            
│   └── eval.py                  
├── slurm/                       # generic SLURM templates
│   ├── unlearn.sh               
│   ├── precompute.sh            
│   └── eval.sh                  
└── prompts/                     # prompts used in experiments
```

### Desired Research Workflow

1. **Prepare Unlearning methods** (`zml/unlearn`): Add code for different unlearning methods there.
2. **Prepare Evaluation methods** (`zml/eval`): Prepare code for different evaluation methods there. Some functions from here should be used during unlearning for live evaluation.
3. **Prepare Precompute methods** (optional) (`zml/precompute`): If we can speed up unlearning, by precomputing some latents or other intermediate results, we add code here.
4. **Prepare thin generic entrypoints** (`scripts/`): These should be thin wrappers that parses arguments call the code in `zml/`.
5. **Prepare SLURM templates** (`slurm/`): These should be generic templetes, one for each type of task. They should call thin entrypoints.
6. **Prepare experiments** (`experiments/`): For each experiment, create a new folder with a config file containing all hyperparameters, dataset info, etc. The experiment config should be in YAML format. Generate new prompt sets if needed.
7. **Run experiments** (`submit_to_athena.sh`): Run experiments on athena cluster. Pass proper slurm script and config file as arguments to the script. The script works by running git pull on athena, so ensure that all necessary content is committed. (for now claude should not submit any jobs by itself, project owners will do it manually)
8. **Collect results** (`pull_from_athena.sh`): Run to download results from athena cluster using rsync. (This requires change, because we used to just sync outputs folder, but now we want to have a folder for each experiment which will have both - config and results)
9. **Evaluate, analyze, iterate**: Look on the results, optionally run additional evaluation scripts, analyze the results, and iterate on the unlearning method or hyperparameters.

### Additional Notes
- You should write clean and maintainable python code and use type hints.
- You should try to extract numeric constants to constants put at the top of the scripts, especially for values that need to be tuned
- You should avoid using too long functions or loops. If some logic is easily separable, extract it to a smaller function. However, be sane and don't force breaking code into functions where it is not natural.
- Inside unlearning scripts we should periodically run evaluation to check the progress.
- Our local computers don't have enough GPU memory (we have no more than 6 GB) to run the experiments, so we need to use the cluster.
- There are three people working on this project.
