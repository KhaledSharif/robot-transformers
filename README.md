# Action Transformers for Robots

This repository contains quickstart code to train and evaluate an Action Chunking Transformer (ACT) to perform
various robot manipulation tasks derived
from [ALOHA](https://github.com/huggingface/gym-aloha), [PushT](https://github.com/huggingface/gym-pusht),
and [xArm](https://github.com/huggingface/gym-xarm).

![out.gif](aloha.gif)

In the ALOHA `TransferCubeTask`, the right arm of a robot needs to pick up a red cube
and place it inside the gripper of the left arm.

View my training & evaluation graphs: https://api.wandb.ai/links/kldsrforg/fzmcvzc7

---

### What is an Action Chunking Transformer (ACT)?

An Action Chunking Transformer is a novel imitation learning algorithm designed to handle the complexities of
fine-grained robotic manipulation tasks. It leverages the strengths of action chunking and the Transformer
architecture to improve the learning and execution of these tasks.

### ACT Key Concepts

1. **Action Chunking**:
    - **Definition**: Action chunking refers to grouping a sequence of actions together and treating them as a single
      unit. Instead of predicting one action at a time, the model predicts a sequence of actions for multiple timesteps.
    - **Purpose**: This reduces the effective horizon of the task, which helps in mitigating the compounding error
      problem. Compounding errors occur when small prediction errors accumulate over time, leading the robot to states
      that are outside the training distribution and causing task failures.
    - **Implementation**: In the context of the Action Chunking Transformer, the policy models the probability
      distribution of a sequence of actions given the current observation.

2. **Transformer Architecture**:
    - **Transformers**: Originally designed for natural language processing tasks, Transformers are effective at
      handling sequence data and capturing long-range dependencies.
    - **Encoder-Decoder Structure**: In this implementation, the Transformer encoder processes the observation inputs
      (including visual data and joint positions), and the Transformer decoder predicts the sequence of actions.
    - **Conditional Variational Autoencoder (CVAE)**: The Action Chunking Transformer uses a CVAE to handle the
      variability in human demonstrations. The CVAE encoder compresses the observed actions and joint positions into a
      latent variable `z`, which the decoder then uses, along with the observations, to predict the sequence of
      actions.

3. **Temporal Ensembling**:
    - **Definition**: Temporal ensembling involves averaging the predictions of overlapping action chunks to produce
      smoother and more accurate trajectories.
    - **Purpose**: This technique addresses the potential issue of abrupt changes between action chunks and ensures
      smoother transitions by incorporating new observations continuously and averaging the predicted actions.
    - **Implementation**: The policy is queried at each timestep, producing overlapping chunks of actions. These
      predicted actions are then combined using an exponential weighting scheme, which prioritizes more recent
      predictions but still takes older ones into account.

### ACT Workflow

1. **Data Collection**:
    - Human demonstrations are collected using a teleoperation system. The joint positions of the leader robot (operated
      by the human) are recorded as the actions, and observations include images from multiple cameras and the joint
      positions of the follower robot.

2. **Training**:
    - The CVAE encoder processes the collected data to learn a latent representation `z`.
    - The Transformer decoder, conditioned on `z` and the current observations, predicts the sequence of future
      actions.
    - The model is trained to minimize the reconstruction loss (difference between predicted and actual actions) and the
      KL-divergence regularization loss to ensure the latent space is well-structured.

3. **Inference**:
    - During execution, the policy generates action sequences based on the current observation and the mean of the prior
      distribution of `z`.
    - Temporal ensembling is applied to combine predictions from overlapping action chunks, ensuring smooth and precise
      motion.

### ACT Advantages

- **Reduction of Compounding Errors**: By predicting action sequences, the effective horizon is reduced, and errors do
  not compound as rapidly.
- **Handling of Non-Markovian Behavior**: Action chunking can manage pauses and other non-Markovian behaviors in human
  demonstrations, improving the robustness of the policy.
- **Smooth and Precise Actions**: Temporal ensembling helps in producing smooth and accurate actions, which are crucial
  for fine-grained manipulation tasks.

### Training

```python
available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
    "xarm": ["XarmLift-v0"],
}
available_datasets_per_env = {
    "aloha": [
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_insertion_scripted",
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
    ],
    "pusht": ["lerobot/pusht"],
    "xarm": [
        "lerobot/xarm_lift_medium",
        "lerobot/xarm_lift_medium_replay",
        "lerobot/xarm_push_medium",
        "lerobot/xarm_push_medium_replay",
    ],
}
```

```bash
python train.py \
   hydra.job.name=act_aloha_sim_transfer_cube_human \
   hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human \
   policy=act \
   policy.use_vae=true \
   env=aloha \
   env.task=AlohaTransferCube-v0 \
   dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
   training.eval_freq=10000 \
   training.log_freq=250 \
   training.offline_steps=100000 \
   training.save_model=true \
   training.save_freq=25000 \
   eval.n_episodes=50 \
   eval.batch_size=50 \
   wandb.enable=true \
   device=cuda
```

```bash
python train.py \
   hydra.job.name=act_aloha_sim_insertion_human \
   hydra.run.dir=outputs/train/act_aloha_sim_insertion_human \
   policy=act \
   policy.use_vae=true \
   env=aloha \
   env.task=AlohaInsertion-v0 \
   dataset_repo_id=lerobot/aloha_sim_insertion_human \
   training.eval_freq=10000 \
   training.log_freq=250 \
   training.offline_steps=100000 \
   training.save_model=true \
   training.save_freq=25000 \
   eval.n_episodes=50 \
   eval.batch_size=50 \
   wandb.enable=true \
   device=cuda
```

### Evaluation

Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```bash
python eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```bash
python eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000 \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found [here](https://huggingface.co/lerobot/diffusion_pusht/tree/main).
