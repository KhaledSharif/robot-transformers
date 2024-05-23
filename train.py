import logging
from copy import deepcopy
from pathlib import Path
import datasets
import hydra
import torch
from datasets import concatenate_datasets
from datasets.utils import disable_progress_bars, enable_progress_bars
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.eval import eval_policy
from utils import (make_optimizer_and_scheduler, update_policy, log_train_info,
                   log_eval_info, calculate_online_sample_weight)


def add_episodes_inplace(
        online_dataset: torch.utils.data.Dataset,
        concat_dataset: torch.utils.data.ConcatDataset,
        sampler: torch.utils.data.WeightedRandomSampler,
        hf_dataset: datasets.Dataset,
        episode_data_index: dict[str, torch.Tensor],
        pc_online_samples: float,
):
    """
    Modifies the online_dataset, concat_dataset, and sampler in place by integrating
    new episodes from hf_dataset into the online_dataset, updating the concatenated
    dataset's structure and adjusting the sampling strategy based on the specified
    percentage of online samples.

    Parameters:
    - online_dataset (torch.utils.data.Dataset): The existing online dataset to be updated.
    - concat_dataset (torch.utils.data.ConcatDataset): The concatenated dataset that combines
      offline and online datasets, used for sampling purposes.
    - sampler (torch.utils.data.WeightedRandomSampler): A sampler that will be updated to
      reflect changes in the dataset sizes and specified sampling weights.
    - hf_dataset (datasets.Dataset): A Hugging Face dataset containing the new episodes to be added.
    - episode_data_index (dict): A dictionary containing two keys ("from" and "to") associated to dataset indices.
      They indicate the start index and end index of each episode in the dataset.
    - pc_online_samples (float): The target percentage of samples that should come from
      the online dataset during sampling operations.

    Raises:
    - AssertionError: If the first episode_id or index in hf_dataset is not 0
    """
    first_episode_idx = hf_dataset.select_columns("episode_index")[0][
        "episode_index"
    ].item()
    last_episode_idx = hf_dataset.select_columns("episode_index")[-1][
        "episode_index"
    ].item()
    first_index = hf_dataset.select_columns("index")[0]["index"].item()
    last_index = hf_dataset.select_columns("index")[-1]["index"].item()
    # sanity check
    assert first_episode_idx == 0, f"{first_episode_idx=} is not 0"
    assert first_index == 0, f"{first_index=} is not 0"
    assert first_index == episode_data_index["from"][first_episode_idx].item()
    assert last_index == episode_data_index["to"][last_episode_idx].item() - 1

    if len(online_dataset) == 0:
        # initialize online dataset
        online_dataset.hf_dataset = hf_dataset
        online_dataset.episode_data_index = episode_data_index
    else:
        # get the starting indices of the new episodes and frames to be added
        start_episode_idx = last_episode_idx + 1
        start_index = last_index + 1

        def shift_indices(episode_index, index):
            # note: we don't shift "frame_index" since it represents the
            # index of the frame in the episode it belongs to
            example = {
                "episode_index": episode_index + start_episode_idx,
                "index": index + start_index,
            }
            return example

        disable_progress_bars()  # map has a tqdm progress bar
        hf_dataset = hf_dataset.map(
            shift_indices, input_columns=["episode_index", "index"]
        )
        enable_progress_bars()

        episode_data_index["from"] += start_index
        episode_data_index["to"] += start_index

        # extend online dataset
        online_dataset.hf_dataset = concatenate_datasets(
            [online_dataset.hf_dataset, hf_dataset]
        )

    # update the concatenated dataset length used during sampling
    concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

    # update the sampling weights for each frame so that online frames get sampled a certain percentage of times
    len_online = len(online_dataset)
    len_offline = len(concat_dataset) - len_online
    weight_offline = 1.0
    weight_online = calculate_online_sample_weight(
        len_offline, len_online, pc_online_samples
    )
    sampler.weights = torch.tensor(
        [weight_offline] * len_offline + [weight_online] * len(online_dataset)
    )

    # update the total number of samples used during sampling
    sampler.num_samples = len(concat_dataset)


@hydra.main(version_base="1.2", config_name="default", config_path="./configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train(cfg: dict, out_dir=None, job_name=None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()

    if cfg.training.online_steps > 0 and cfg.eval.batch_size > 1:
        logging.warning("eval.batch_size > 1 not supported for online training steps")

    # Check device is available
    get_safe_torch_device(cfg.device, log=True)

    # Enables cuDNN benchmark mode for better performance on certain input shapes
    torch.backends.cudnn.benchmark = True

    # Allows TensorFloat32 (TF32) on Tensor Core for better performance on certain operations
    torch.backends.cuda.matmul.allow_tf32 = True

    set_global_seed(cfg.seed)

    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg)

    logging.info("make_env")
    eval_env = make_env(cfg)

    logging.info("make_policy")
    policy = make_policy(hydra_cfg=cfg, dataset_stats=offline_dataset.stats)

    # Create optimizer and scheduler
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    # log metrics to terminal and wandb
    logger = Logger(out_dir, job_name, cfg)

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(
        f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})"
    )
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(
        f"{offline_dataset.num_samples=} ({format_big_number(offline_dataset.num_samples)})"
    )
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops
    # It evaluates the policy on the evaluation environment, logs the evaluation results,
    # and saves a model checkpoint at specified intervals (cfg.training.eval_freq and
    # cfg.training.save_freq, respectively)
    def evaluate_and_checkpoint_if_needed(step):
        if step % cfg.training.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            eval_info = eval_policy(
                eval_env,
                policy,
                cfg.eval.n_episodes,
                video_dir=Path(out_dir) / "eval",
                max_episodes_rendered=4,
                start_seed=cfg.seed,
            )
            log_eval_info(
                logger, eval_info["aggregated"], step, cfg, offline_dataset, is_offline
            )
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.training.save_model and step % cfg.training.save_freq == 0:
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_model(
                policy,
                identifier=str(step).zfill(
                    max(
                        6,
                        len(
                            str(cfg.training.offline_steps + cfg.training.online_steps)
                        ),
                    )
                ),
            )
            logging.info("Resume training")

    # create dataloader for offline training
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=4,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=cfg.device != "cpu",
        drop_last=False,
    )

    # Create an iterator dl_iter by applying cycle to the dataloader,
    # which allows iterating over the dataset indefinitely
    dl_iter = cycle(dataloader)

    # Set policy to training mode
    policy.train()

    # number of policy update (forward + backward + optim)
    step = 0

    is_offline = True
    for offline_step in range(cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")
        batch = next(dl_iter)

        # moves each tensor in the batch dictionary to the specified device
        for key in batch:
            batch[key] = batch[key].to(cfg.device, non_blocking=True)

        # performs a single policy update step by calling the update_policy function,
        # which computes the loss, performs backpropagation, & updates policy params
        train_info = update_policy(
            policy, batch, optimizer, cfg.training.grad_clip_norm, lr_scheduler
        )

        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_offline)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1)

        step += 1

    # create an env dedicated to online episodes collection from policy rollout
    online_training_env = make_env(cfg, n_envs=1)

    # create an empty online dataset similar to offline dataset
    online_dataset = deepcopy(offline_dataset)
    online_dataset.hf_dataset = {}
    online_dataset.episode_data_index = {}

    # create dataloader for online training
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    weights = [1.0] * len(concat_dataset)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(concat_dataset), replacement=True
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        num_workers=4,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        pin_memory=cfg.device != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    online_step = 0
    is_offline = False
    for env_step in range(cfg.training.online_steps):
        if env_step == 0:
            logging.info("Start online training by interacting with environment")

        policy.eval()
        with torch.no_grad():
            eval_info = eval_policy(
                online_training_env,
                policy,
                n_episodes=1,
                return_episode_data=True,
                start_seed=cfg.training.online_env_seed,
                enable_progbar=True,
            )

        add_episodes_inplace(
            online_dataset,
            concat_dataset,
            sampler,
            hf_dataset=eval_info["episodes"]["hf_dataset"],
            episode_data_index=eval_info["episodes"]["episode_data_index"],
            pc_online_samples=cfg.training.online_sampling_ratio,
        )

        policy.train()
        for _ in range(cfg.training.online_steps_between_rollouts):
            batch = next(dl_iter)

            for key in batch:
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

            train_info = update_policy(
                policy, batch, optimizer, cfg.training.grad_clip_norm, lr_scheduler
            )

            if step % cfg.training.log_freq == 0:
                log_train_info(
                    logger, train_info, step, cfg, online_dataset, is_offline
                )

            # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
            # so we pass in step + 1.
            evaluate_and_checkpoint_if_needed(step + 1)

            step += 1
            online_step += 1

    eval_env.close()
    online_training_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    train_cli()
