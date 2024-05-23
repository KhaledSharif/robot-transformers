import logging
import time
import torch
from lerobot.common.logger import Logger
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.utils.utils import (
    format_big_number,
)


def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "act":
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not n.startswith("backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if n.startswith("backbone") and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_params_dicts,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        lr_scheduler = None
    elif cfg.policy.name == "diffusion":
        optimizer = torch.optim.Adam(
            policy.diffusion.parameters(),
            cfg.training.lr,
            cfg.training.adam_betas,
            cfg.training.adam_eps,
            cfg.training.adam_weight_decay,
        )
        assert (
                cfg.training.online_steps == 0
        ), "Diffusion Policy does not handle online training."
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    elif policy.name == "tdmpc":
        optimizer = torch.optim.Adam(policy.parameters(), cfg.training.lr)
        lr_scheduler = None
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler


def update_policy(policy, batch, optimizer, grad_clip_norm, lr_scheduler=None):
    """Returns a dictionary of items for logging."""
    start_time = time.time()
    policy.train()
    output_dict = policy.forward(batch)
    # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    loss = output_dict["loss"]
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if isinstance(policy, PolicyWithUpdate):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    info = {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.time() - start_time,
        **{k: v for k, v in output_dict.items() if k != "loss"},
    }

    return info


def log_train_info(logger: Logger, info, step, cfg, dataset, is_offline):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        f"grdn:{grad_norm:.3f}",
        f"lr:{lr:0.1e}",
        # in seconds
        f"updt_s:{update_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_offline"] = is_offline

    logger.log_dict(info, step, mode="train")


def log_eval_info(logger, info, step, cfg, dataset, is_offline):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"∑rwrd:{avg_sum_reward:.3f}",
        f"success:{pc_success:.1f}%",
        f"eval_s:{eval_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_offline"] = is_offline

    logger.log_dict(info, step, mode="eval")


def calculate_online_sample_weight(n_off: int, n_on: int, pc_on: float):
    """
    Calculate the sampling weight to be assigned to samples so that a specified percentage of the batch comes from online dataset (on average).

    Parameters:
    - n_off (int): Number of offline samples, each with a sampling weight of 1.
    - n_on (int): Number of online samples.
    - pc_on (float): Desired percentage of online samples in decimal form (e.g., 50% as 0.5).

    The total weight of offline samples is n_off * 1.0.
    The total weight of offline samples is n_on * w.
    The total combined weight of all samples is n_off + n_on * w.
    The fraction of the weight that is online is n_on * w / (n_off + n_on * w).
    We want this fraction to equal pc_on, so we set up the equation n_on * w / (n_off + n_on * w) = pc_on.
    The solution is w = - (n_off * pc_on) / (n_on * (pc_on - 1))
    """
    assert 0.0 <= pc_on <= 1.0
    return -(n_off * pc_on) / (n_on * (pc_on - 1))
