import argparse
import json
import logging
from datetime import datetime as dt
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from utils import eval_policy


def eval_cli():
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        eval(hydra_cfg_path=args.config, config_overrides=args.overrides)
    else:
        try:
            pretrained_policy_path = Path(
                snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
            )
        except (HFValidationError, RepositoryNotFoundError) as e:
            if isinstance(e, HFValidationError):
                error_message = (
                    "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
                )
            else:
                error_message = (
                    "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
                )

            logging.warning(f"{error_message} Treating it as a local directory.")
            pretrained_policy_path = Path(args.pretrained_policy_name_or_path)
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        evaluate(pretrained_policy_path=pretrained_policy_path, config_overrides=args.overrides)


def evaluate(
        pretrained_policy_path: str | None = None,
        hydra_cfg_path: str | None = None,
        config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if hydra_cfg_path is None:
        hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)
    out_dir = (
        f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"
    )

    if out_dir is None:
        raise NotImplementedError()

    # Check device is available
    get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)
    policy.eval()

    info = eval_policy(
        env,
        policy,
        hydra_cfg.eval.n_episodes,
        max_episodes_rendered=10,
        video_dir=Path(out_dir) / "eval",
        start_seed=hydra_cfg.seed,
        enable_progbar=True,
        enable_inner_progbar=True,
    )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    eval_cli()
