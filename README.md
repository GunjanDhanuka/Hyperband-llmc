# Hyperband: Bandit-based Hyperparameter Optimization for GPT-2 [LLM.C]
This is a Python implementation of the Hyperband algorithm proposed in [this](https://arxiv.org/abs/1603.06560) paper for hyperparameter optimization which uses Multi-Armed bandits with a hedging between exploration and exploitation, testing out a much larger number of configurations in the same compute budget compared to random search/grid search. The main aim of this project was to perform hyperparameter optimization on the GPT-2 Model from [LLM.C](https://github.com/karpathy/llm.c) implementation by Andrej Karpathy.

## Running the code
1. Copy the `hyperband.py` file to the directory where you have built the executables for GPT-2 from LLM.C repo.
2. If you are enabling Cosine Decay for LR, it is advised to modify this [line](https://github.com/karpathy/llm.c/blob/master/llmc/schedulers.h#L33) to:
`float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (20000 - scheduler->warmup_iterations);`, where 20000 is the number of steps I want to train the final model for. This simulates the LR decay in a way that is exactly similar to how you would do if you ran the training for one full epoch (basically the `-x` flag in llm.c).
3. Run the hyperparameter tuning code using `python hyperband.py`.

You might want to tweak the `max_iter` parameter based on your total compute budget and time constraints. The `total_iters.py` file contains the code to calculate the total number of iterations, and cost values for a `g5.xlarge` instance on AWS EC2. You can modify that according to your requirement.

The current code supports tuning all kinds of hyperparameter but by default does only Learning Rate tuning, as that was the scope of the project. You can add or remove parameters from the config as per your need. Also, before setting bounds for random sampling of hyperparameters, try to run the smaller variant of model on multiple distant values, to get a rough estimate of what range would be best for that hyperparameter.

## Features
1. Supports saving and resuming from checkpoints.
2. Deletes old checkpoints when the configuration is evicted.
3. Detailed logging of the training progress.
4. Saves the best configuration in the end.
5. Flexible to add more hyperparameters.

> Note: This code was written for an assignment in [10605: ML with Large Datasets](https://10605.github.io/) (Fall'24) at Carnegie Mellon University, and was made public after approval from the course staff.
