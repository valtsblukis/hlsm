# HLSM
This is the code repository for the paper [A Persistent Spatial Semantic Representation for High-level Natural Language Instruction Execution](https://arxiv.org/abs/2107.05612).

## Setup

Tested on Ubuntu 20.04.

### Setup Python Environment
```
conda env create -f hlsm-alfred.yml
conda activate hlsm-alfred
```

### Setup Workspace
1. Create a workspace directory somewhere on your system to store models, checkpoints, data, alfred source, etc.
Collecting training data requires ~600GB of space. SSD preferred for faster training.
```
mkdir <workspace dir>
```

2. Update the WS_DIR variable in init.sh to point to `<workspace_dir>`.  

3. Clone ALFRED into sub-directory `alfred_src` in the workspace:
```
cd <workspace dir>
mkdir alfred_src
cd alfred_src
git clone https://github.com/askforalfred/alfred.git
```

4. Define environment variables. **Do this before running every script in this repo.**
```
source init.sh
```

5. (Optional) Download pre-trained models from [this Google Drive link](https://drive.google.com/drive/folders/1PlZGHQLAirFoX9lmrz32S3C1SjNVmTq-?usp=sharing) and save them into `<workspace_dir>/models`.


### Configuration Files
Most scripts in this repo are parameterized by a single argument that specifies a json
configuration file stored in `lgp/experiment_definitions`, excluding the .json file extension.
To change hyperparameters, datasets, or models, you can modify the existing .json configurations
or create your own. The configuration files support a special @include directive that allows recursively including other
configuration files (useful to share parameters among multiple configurations).

## Collect Training Data
Training data consists of two datasets extracted from oracle rollouts in the ALFRED environment.
The subgoal dataset consists of examples of semantic maps at the start of each navigation+manipulation sequence
labelled with subgoals. The navigation dataset consists of RGB images, ground-truth depth and segmentation, and
2D affordance feature maps at every timestep labelled with navigation goal poses.

The following command will execute the oracle actions on every training environment in ALFRED,
and store the data into `<workspace_dir>/data/rollouts`. It will take a few days to run.
Incase the process is interrupted, run it again to resume data collection.
```
python main/collect_universal_rollouts.py alfred/collect_universal_rollouts
```

## Training
Trained models are stored in <workspace_dir>/models.
If you downloaded the models above, you can skip these training steps and proceed to evaluation.

1. Train the high-level controller (subgoal model) for 6 epochs:
```
python main/train_supervised.py alfred/train_subgoal_model
```

2. Train the low-level controller's navigation model for 6 epochs:
```
python main/train_supervised.py alfred/train_navigation_model
```

3. Train the semantic segmentation model for 5 epochs:
```
python main/train_supervised.py alfred/train_segmentation_model
```

4. Train the depth prediction model for 4 epochs:
```
python main/train_supervised.py alfred/train_depth_model
```

Tensorboard summaries are written to `workspace_dir/data/runs`.

## Evaluation
`main/rollout_and_evaluate.py` is the main evaluation script.
Call it with different configurations to evaluate on the different data splits.

To evaluate on valid_unseen, run:
```
python main/rollout_and_evaluate.py alfred/eval/hlsm_full/eval_hlsm_valid_unseen
```

To evaluate on valid_seen, run:
```
python main/rollout_and_evaluate.py alfred/eval/hlsm_full/eval_hlsm_valid_seen
```

To evaluate on both test splits and collect traces for leaderboard, run:
```
python main/rollout_and_evaluate.py alfred/eval/hlsm_full/eval_hlsm_test
```

Explore the other configurations available at `experiment_definitions/alfred/eval`.


**Expected results**:

| Data split      | SR          | GC          |
| --------------- | ----------- | ----------- |
| valid_unseen    | 19.48%      | 32.5%       |
| valid_seen      | 29.63%      | 38.7%       |
| tests_unseen    | 20.27%      | 30.3%       |
| tests_seen      | 29.94%      | 41.2%       |
