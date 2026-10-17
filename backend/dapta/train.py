from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import REQUIRED_SPLITS, sha256, to_training_rows
from .validation import require_valid_receipt_dataset


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Training uploads here; only the publish stage promotes an adapter that passed frozen TEST.
CANDIDATE_PREFIX = "candidates"
# Qwen3-VL spends one visual token per 32x32 pixels (patch 16, merge 2). The processor
# default allows 16.7M pixels (~16k tokens) per image; 1280 tokens keeps receipt text
# legible while bounding memory and step time on a 48 GB A40.
DEFAULT_MAX_PIXELS = 1280 * 32 * 32
# Language-model projections only: the Qwen3-VL vision tower names its layers qkv, proj
# and linear_fc1/2, so these names leave the vision encoder frozen.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TrainOptions:
    base_model: str = MODEL_ID
    epochs: float = 2
    learning_rate: float = 1e-4
    max_steps: int = -1
    max_pixels: int = DEFAULT_MAX_PIXELS
    lora_rank: int = 16
    # False only for the local CPU check: no 4-bit quantization, no GPU, no upload.
    gpu: bool = True


def train(
    dataset_dir: Path,
    output_dir: Path,
    skill_id: str,
    version: str,
    *,
    contract_dir: Path,
    source_manifest: Path,
    options: TrainOptions = TrainOptions(),
) -> Path:
    report = require_valid_receipt_dataset(
        dataset_dir,
        contract_dir=contract_dir,
        source_manifest=source_manifest,
    )

    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    if options.gpu and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available (torch {torch.__version__}, built for CUDA "
            f"{torch.version.cuda}); check the worker's NVIDIA driver"
        )
    if options.max_pixels < 32 * 32:
        raise ValueError("max_pixels must allow at least one visual token")

    split_counts = report["counts"]

    def load_split(split: str) -> Dataset:
        rows = to_training_rows(dataset_dir, split)
        for row in rows:
            loaded = []
            for path in row["images"]:
                with Image.open(path) as image:
                    loaded.append(image.convert("RGB"))
            row["images"] = loaded
        return Dataset.from_list(rows)

    train_dataset = load_split("train")
    dev_dataset = load_split("dev")
    adapter_dir = output_dir / skill_id / version
    adapter_dir.mkdir(parents=True, exist_ok=False)
    (adapter_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(options.base_model, token=token)
    image_processor = processor.image_processor
    min_pixels = min(image_processor.size.get("shortest_edge", 32 * 32), options.max_pixels)
    image_processor.size = {"shortest_edge": min_pixels, "longest_edge": options.max_pixels}

    load_options = {"token": token, "attn_implementation": "sdpa"}
    if options.gpu:
        load_options.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
    else:
        load_options.update(dtype=torch.float32, device_map={"": "cpu"})
    model = Qwen3VLForConditionalGeneration.from_pretrained(options.base_model, **load_options)
    model.config.use_cache = False
    if options.gpu:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=options.lora_rank,
        lora_alpha=2 * options.lora_rank,
        lora_dropout=0.05,
        target_modules=list(LORA_TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(adapter_dir / "checkpoints"),
        num_train_epochs=options.epochs,
        max_steps=options.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=options.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=options.gpu,
        fp16=False,
        use_cpu=not options.gpu,
        gradient_checkpointing=options.gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        max_length=None,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor,
        peft_config=lora,
    )
    result = trainer.train()
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    metadata = {
        "skill_id": skill_id,
        "version": version,
        "base_model": options.base_model,
        "dataset": {
            "counts": split_counts,
            "sha256": {split: sha256(dataset_dir / f"{split}.jsonl") for split in REQUIRED_SPLITS},
        },
        "method": "4-bit QLoRA" if options.gpu else "LoRA (CPU check)",
        "lora": {
            "r": lora.r,
            "alpha": lora.lora_alpha,
            "dropout": lora.lora_dropout,
            "targets": sorted(lora.target_modules),
        },
        "training": {
            **asdict(options),
            "batch_size": 1,
            "gradient_accumulation": 8,
            "seed": 42,
        },
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
        },
        "train_metrics": result.metrics,
    }
    (adapter_dir / "dapta.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    repo_id = os.environ.get("HF_ADAPTER_REPO")
    if repo_id and options.gpu:
        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            folder_path=adapter_dir,
            path_in_repo=f"{CANDIDATE_PREFIX}/{skill_id}/{version}",
            commit_message=f"Upload candidate {skill_id} {version}",
            # Checkpoints hold optimizer state; only the final adapter is uploaded.
            ignore_patterns=["checkpoints/*", "checkpoints/**"],
        )
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapters"))
    parser.add_argument("--skill", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/adapter-assets/skills/receipt-to-expense"),
    )
    parser.add_argument(
        "--sources", type=Path, default=Path("research/adapter-assets/sources.lock.json")
    )
    defaults = TrainOptions()
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--epochs", type=float, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--max-pixels", type=int, default=defaults.max_pixels)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument(
        "--cpu-check",
        action="store_true",
        help="run the real pipeline on CPU without quantization or upload (use a tiny model)",
    )
    args = parser.parse_args()
    path = train(
        args.dataset,
        args.output,
        args.skill,
        args.version,
        contract_dir=args.contract,
        source_manifest=args.sources,
        options=TrainOptions(
            base_model=args.base_model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            max_pixels=args.max_pixels,
            lora_rank=args.lora_rank,
            gpu=not args.cpu_check,
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
