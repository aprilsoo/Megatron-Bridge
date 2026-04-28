#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic Training Script for LLM and diffusion models

This script works with any model family that uses GPT-style training
(Llama, Gemma, Qwen, GPT, etc.) and with diffusion models (e.g. FLUX, WAN). It dynamically loads recipes and supports
CLI overrides. The --dataset flag selects the dataset type and automatically
infers pretrain vs finetune mode.

Usage:
    Pretrain (mock data):
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe llama32_1b_pretrain_config \\
            --dataset llm-pretrain-mock

    Pretrain (real data):
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe llama32_1b_pretrain_config \\
            --dataset llm-pretrain \\
            'dataset.blend=[[/data/my_dataset_text_document],null]'

    Finetune (SQuAD, default):
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe llama32_1b_sft_config \\
            --dataset llm-finetune

    Finetune (GSM8K):
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe llama32_1b_sft_config \\
            --dataset llm-finetune \\
            dataset.dataset_name=gsm8k

    Finetune (user-supplied JSONL):
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe llama32_1b_sft_config \\
            --dataset llm-finetune-preloaded \\
            dataset.dataset_root=/data/my_finetune_data

    Diffusion pretrain:
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe wan_1_3B_pretrain_config \
            --step_func wan_step \
            dataset.path=/data/energon

    Diffusion SFT (full finetuning):
        uv run torchrun --nproc_per_node=8 run_recipe.py \
            --recipe wan_1_3B_sft_config \
            --step_func wan_step
            dataset.path=/data/energon

    VLM with HF dataset:
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe qwen3_vl_8b_peft_config \\
            --dataset vlm-hf \\
            --step_func qwen3_vl_step \\
            dataset.maker_name=cord_v2 \\
            dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct \\
            checkpoint.pretrained_checkpoint=/path/to/checkpoint

    VLM with Energon dataset:
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe qwen3_vl_8b_peft_energon_config \\
            --dataset vlm-energon \\
            --step_func qwen3_vl_step \\
            dataset.path=/data/energon \\
            checkpoint.pretrained_checkpoint=/path/to/checkpoint

    VLM with preloaded JSON:
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe qwen3_vl_8b_peft_config \\
            --dataset vlm-preloaded \\
            --step_func qwen3_vl_step \\
            dataset.train_data_path=/data/vlm_train.json \\
            dataset.image_folder=/data/vlm_images \\
            dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct \\
            checkpoint.pretrained_checkpoint=/path/to/checkpoint

    With CLI overrides (Hydra-style, works for any config field):
        uv run torchrun --nproc_per_node=8 run_recipe.py \\
            --recipe llama32_1b_pretrain_config \\
            --dataset llm-pretrain-mock \\
            train.train_iters=5000 \\
            optimizer.lr=0.0003

Recipe Arguments:
    Generic scripts call recipes with no arguments: recipe().

    If you need to pass arguments to the recipe constructor
    (e.g., custom parallelism at build time), create a custom script.
"""

import argparse
import inspect
import sys
from typing import Callable

import megatron.bridge.recipes as recipes

# Diffusion forward steps: use class instances so they can be passed as forward_step_func
from megatron.bridge.diffusion.models.flux.flux_step import FluxForwardStep
from megatron.bridge.diffusion.models.wan.wan_step import WanForwardStep
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.recipes.utils.dataset_utils import (
    DATASET_TYPES,
    apply_dataset_override,
    infer_mode_from_dataset,
)
from megatron.bridge.training.audio_lm_step import forward_step as audio_lm_forward_step
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step
from megatron.bridge.training.llava_step import forward_step as llava_forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step


STEP_FUNCTIONS: dict[str, Callable] = {
    "audio_lm_step": audio_lm_forward_step,
    "gpt_step": gpt_forward_step,
    "vlm_step": vlm_forward_step,
    "qwen3_vl_step": qwen3_vl_forward_step,
    "llava_step": llava_forward_step,
    "flux_step": FluxForwardStep,
    "wan_step": WanForwardStep,
}

TRAIN_FUNCTIONS = {
    "pretrain": pretrain,
    "finetune": finetune,
}

ERR_UNKNOWN_STEP = "Unknown step type: {step_type}. Choose from: {choices}"
ERR_INFER_MODE_FAILED = (
    "Unable to infer training mode. "
    "Pass --dataset to specify the dataset type, or include 'pretrain' or 'finetune' "
    "(or 'sft'/'peft') in the recipe name."
)


def parse_args() -> tuple[argparse.Namespace, list[str], set[str]]:
    """Parse command-line arguments.

    Returns:
        (args, cli_overrides, cli_specified): parsed args, Hydra overrides, and set of CLI-specified arg names
    """
    parser = argparse.ArgumentParser(
        description="Generic training script for LLM and diffusion models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="YAML config file. _args section provides CLI args (recipe, dataset, etc.), "
             "remaining sections are Megatron-Bridge config overrides",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        required=False,
        default=None,
        help="Recipe function name (e.g., llama32_1b_pretrain_config, gemma3_1b_sft_config, gemma3_1b_peft_config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=DATASET_TYPES,
        help=(
            "Dataset type. Training mode (pretrain/finetune) is inferred from this.\n"
            "LLM datasets:\n"
            "  llm-pretrain           GPT pretrain data (set dataset.blend=<path>)\n"
            "  llm-pretrain-mock      Mock pretrain data for testing\n"
            "  llm-finetune           HF finetune dataset (set dataset.dataset_name=squad|gsm8k|openmathinstruct2)\n"
            "  llm-finetune-preloaded User-supplied JSONL (set dataset.dataset_root=<path>)\n"
            "VLM datasets:\n"
            "  vlm-energon            Energon multimodal (set dataset.path=<path>)\n"
            "  vlm-hf                 HF VLM dataset (set dataset.maker_name=<name>)\n"
            "  vlm-preloaded          User-supplied VLM JSON (set dataset.train_data_path=<path>)"
        ),
    )
    parser.add_argument(
        "--step_func",
        type=str,
        default="gpt_step",
        choices=sorted(STEP_FUNCTIONS.keys()),
        help="Step function: gpt_step (text-only), vlm_step (vision-language), llava_step (LLaVA), "
        "flux_step (FLUX diffusion), wan_step (WAN diffusion, hyperparameters selected by --mode/recipe name)",
    )
    parser.add_argument(
        "--peft_scheme",
        type=str,
        default=None,
        help="PEFT scheme to use: 'lora', 'dora', or None.",
    )
    parser.add_argument(
        "--packed_sequence",
        action="store_true",
        default=False,
        help="Enable packed sequence training (default: False)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path to model directory. "
        "Use a local path for more stable multinode training.",
    )
    args, cli_overrides = parser.parse_known_args()

    # Detect which parameters were explicitly specified on the command line
    # Use more robust detection: check exact argv matches with = for long options
    cli_specified = set()
    argv_set = set(sys.argv[1:])  # Exclude script name

    for action in parser._actions:
        if action.dest == 'help':
            continue

        # Check each option string (e.g., '--recipe', '--dataset')
        for opt in action.option_strings:
            # Exact match (flag alone: '--recipe' followed by value)
            if opt in argv_set:
                cli_specified.add(action.dest)
                break
            # Match with = (e.g., '--recipe=llama32')
            if any(arg.startswith(f"{opt}=") for arg in argv_set):
                cli_specified.add(action.dest)
                break

    return args, cli_overrides, cli_specified


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    hf_path: str | None = None,
) -> ConfigContainer:
    """
    Load recipe by name from megatron.bridge.recipes.

    Args:
        recipe_name: Full recipe function name (e.g., 'llama32_1b_pretrain_config')
        peft_scheme: PEFT scheme to use ('lora', 'dora', or None)
        packed_sequence: Enable packed sequence training (default: False)
        seq_length: Sequence length for training (optional)
        hf_path: HuggingFace model ID or local path to model directory (optional)

    Returns:
        ConfigContainer from calling the recipe

    Raises:
        AttributeError: If recipe not found
    """
    if not hasattr(recipes, recipe_name):
        raise AttributeError(
            f"Recipe '{recipe_name}' not found in megatron.bridge.recipes.\n"
            f"Make sure the recipe name is correct and the recipe is exported in its family __init__.py.\n"
            f"Example recipe names: llama32_1b_pretrain_config, gemma3_1b_pretrain_config, qwen3_8b_pretrain_config"
        )

    config_builder = getattr(recipes, recipe_name)

    # Inspect the recipe's signature to determine which arguments it accepts
    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        accepts_peft = "peft" in params or has_var_keyword
        accepts_packed_sequence = "packed_sequence" in params or has_var_keyword
        accepts_seq_length = "seq_length" in params or has_var_keyword
        accepts_hf_path = "hf_path" in params or has_var_keyword
    except (ValueError, TypeError):
        # If signature inspection fails, fallback conservatively
        accepts_peft = True  # peft is widely supported, try passing it
        accepts_packed_sequence = False  # new parameter, don't pass if unsure
        accepts_seq_length = False  # new parameter, don't pass if unsure
        accepts_hf_path = False  # model-specific, don't pass if unsure

    # Build kwargs dynamically based on what the recipe accepts
    kwargs = {}
    if accepts_peft:
        kwargs["peft"] = peft_scheme
    if accepts_packed_sequence and packed_sequence:
        kwargs["packed_sequence"] = packed_sequence
    if accepts_seq_length and seq_length is not None:
        kwargs["seq_length"] = seq_length
    if accepts_hf_path and hf_path is not None:
        kwargs["hf_path"] = hf_path

    try:
        return config_builder(**kwargs)
    except TypeError:
        # Fallback if the kwargs are not accepted despite signature inspection
        return config_builder()


def load_forward_step(step_type: str, mode: str | None = None) -> Callable:
    """Load forward_step function based on the requested step type."""
    step_key = step_type.lower()
    if step_key not in STEP_FUNCTIONS:
        raise ValueError(ERR_UNKNOWN_STEP.format(step_type=step_type, choices=", ".join(STEP_FUNCTIONS)))
    step = STEP_FUNCTIONS[step_key]
    if inspect.isclass(step):
        if "mode" in inspect.signature(step.__init__).parameters:
            return step(mode=mode)
        return step()
    return step


def infer_train_mode(recipe_name: str) -> str:
    """Infer training mode from the recipe name (fallback when --dataset is not passed)."""
    lowered = recipe_name.lower()
    has_pretrain = "pretrain" in lowered
    has_finetune = "finetune" in lowered or "sft" in lowered or "peft" in lowered
    if has_pretrain ^ has_finetune:
        return "pretrain" if has_pretrain else "finetune"
    raise ValueError(ERR_INFER_MODE_FAILED)


def load_config_file(config_file: str) -> tuple[dict, dict]:
    """Load config file and separate _args (CLI parameters) from Megatron-Bridge overrides.

    Args:
        config_file: Path to configuration file

    Returns:
        (args_dict, override_dict): CLI arguments dict and config overrides dict

    Raises:
        FileNotFoundError: File not found
        ValueError: YAML format error or missing required fields
    """
    import os
    import yaml
    from pathlib import Path

    # Path traversal protection: resolve to absolute path
    config_path = Path(config_file).resolve()

    # Exception handling: file not found
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Exception handling: YAML parsing error
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_file}:\n{e}")

    # Type check
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{config_file} must contain a YAML dictionary, got {type(cfg).__name__}"
        )

    # Extract _args
    args_dict = cfg.pop('_args', {})
    if not args_dict:
        raise ValueError(
            f"Missing required '_args' section in {config_file}.\n"
            f"Please add CLI arguments like:\n"
            f"_args:\n"
            f"  recipe: <recipe_name>\n"
            f"  dataset: <dataset_type>\n"
            f"  step_func: <step_function>"
        )

    # Validate required fields
    required = ['recipe']  # recipe is the only truly required field
    missing = [k for k in required if k not in args_dict]
    if missing:
        raise ValueError(
            f"Missing required fields in '_args' section of {config_file}: {missing}"
        )

    # Remove all reserved sections starting with _
    for key in list(cfg.keys()):
        if key.startswith('_'):
            cfg.pop(key)

    return args_dict, cfg  # cfg is pure override dict


def main() -> None:
    """Run GPT training (pretrain or finetune)."""
    args, cli_overrides, cli_specified = parse_args()

    override_dict = None
    if args.config_file:
        # Load config file (will raise exceptions handled at top level)
        args_from_file, override_dict = load_config_file(args.config_file)

        # Merge parameters: priority = CLI explicit > config file > argparse default
        # Only update parameters that were NOT explicitly specified on CLI
        known_args = vars(args).keys()
        for key, value in args_from_file.items():
            if key in known_args and key not in cli_specified:
                setattr(args, key, value)

    # Final check for required parameters
    if not args.recipe:
        if args.config_file:
            sys.exit(f"Error: Missing 'recipe' in config file '{args.config_file}'")
        else:
            sys.exit("Error: --recipe is required (via CLI or --config-file)")

    # step_func already has default value 'gpt_step' from argparse, no need to set again

    config: ConfigContainer = load_recipe(
        args.recipe,
        args.peft_scheme,
        args.packed_sequence,
        args.seq_length,
        args.hf_path,
    )

    if args.dataset is not None:
        mode = infer_mode_from_dataset(args.dataset)
        config = apply_dataset_override(
            config,
            dataset_type=args.dataset,
            packed_sequence=args.packed_sequence,
            seq_length=args.seq_length,
            cli_overrides=cli_overrides,
        )
    else:
        mode = infer_train_mode(args.recipe)

    config = process_config_with_overrides(
        config,
        config_dict=override_dict,
        cli_overrides=cli_overrides or None,
    )

    # Ensure dataset.seq_length and model.seq_length stay in sync after CLI overrides
    if (
        hasattr(config, "model")
        and config.model is not None
        and hasattr(config, "dataset")
        and config.dataset is not None
    ):
        if hasattr(config.dataset, "seq_length") and config.model.seq_length != config.dataset.seq_length:
            config.model.seq_length = config.dataset.seq_length

    forward_step = load_forward_step(args.step_func, mode=mode)
    train_func = TRAIN_FUNCTIONS[mode]
    train_func(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
