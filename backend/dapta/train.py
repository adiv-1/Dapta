from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import REQUIRED_SPLITS, sha256, to_training_rows
from .validation import require_valid_receipt_dataset


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def train(
    dataset_dir: Path,
    output_dir: Path,
    skill_id: str,
    version: str,
    *,
    contract_dir: Path,
    source_manifest: Path,
) -> Path:
    report = require_valid_receipt_dataset(
        dataset_dir,
        contract_dir=contract_dir,
        source_manifest=source_manifest,
    )

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

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

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        quantization_config=quantization,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(adapter_dir / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
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
        "base_model": MODEL_ID,
        "dataset": {
            "counts": split_counts,
            "sha256": {split: sha256(dataset_dir / f"{split}.jsonl") for split in REQUIRED_SPLITS},
        },
        "method": "4-bit QLoRA",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": sorted(lora.target_modules)},
        "training": {"epochs": 2, "batch_size": 1, "gradient_accumulation": 8, "seed": 42},
        "train_metrics": result.metrics,
    }
    (adapter_dir / "dapta.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    repo_id = os.environ.get("HF_ADAPTER_REPO")
    if repo_id:
        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
            repo_id=repo_id,
            folder_path=adapter_dir,
            path_in_repo=f"{skill_id}/{version}",
            commit_message=f"Publish {skill_id} {version}",
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
    args = parser.parse_args()
    path = train(
        args.dataset,
        args.output,
        args.skill,
        args.version,
        contract_dir=args.contract,
        source_manifest=args.sources,
    )
    print(path)


if __name__ == "__main__":
    main()
