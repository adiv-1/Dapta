from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from .data import REQUIRED_SPLITS, read_jsonl, sha256, to_training_rows
from .validation import require_valid_receipt_dataset


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MIN_IMAGE_PIXELS = 16 * 28 * 28
MAX_IMAGE_PIXELS = 576 * 28 * 28
MAX_SEQUENCE_LENGTH = 2048


def train(
    dataset_dir: Path,
    output_dir: Path,
    skill_id: str,
    version: str,
    *,
    contract_dir: Path,
    source_manifest: Path,
    epochs: float = 2,
    max_steps: int | None = None,
    smoke_inference: bool = False,
) -> Path:
    if skill_id == "receipt-to-expense":
        report = require_valid_receipt_dataset(
            dataset_dir,
            contract_dir=contract_dir,
            source_manifest=source_manifest,
        )
    elif skill_id == "csv-to-dashboard":
        from .dashboards import validate_dashboard_dataset

        report = validate_dashboard_dataset(
            dataset_dir,
            contract_dir=contract_dir,
            source_manifest=source_manifest,
        )
        if not report["passed"]:
            raise ValueError(f"dashboard dataset failed validation: {report['errors'][0]}")
    else:
        raise ValueError(f"unsupported skill: {skill_id}")

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
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
            if loaded:
                row["images"] = loaded
            else:
                row.pop("images")
        return Dataset.from_list(rows)

    train_dataset = load_split("train")
    dev_dataset = load_split("dev")
    adapter_dir = output_dir / skill_id / version
    adapter_dir.mkdir(parents=True, exist_ok=False)
    (adapter_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ.get("HF_TOKEN"),
        min_pixels=MIN_IMAGE_PIXELS,
        max_pixels=MAX_IMAGE_PIXELS,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ.get("HF_TOKEN"),
        quantization_config=quantization,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    smoke_row = read_jsonl(dataset_dir / "dev.jsonl")[0] if smoke_inference else None
    base_result = (
        _generate(model, processor, smoke_row, dataset_dir, Image, torch)
        if smoke_row is not None
        else None
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
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        completion_only_loss=True,
        max_length=MAX_SEQUENCE_LENGTH,
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
    smoke_result = None
    if smoke_row is not None:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=os.environ.get("HF_TOKEN"),
            quantization_config=quantization,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        reloaded = PeftModel.from_pretrained(base_model, adapter_dir)
        adapted_result = _generate(reloaded, processor, smoke_row, dataset_dir, Image, torch)
        smoke_result = {
            "case_id": smoke_row["id"],
            "gold": smoke_row["completion"],
            "base": base_result,
            "adapted": adapted_result,
            "adapter_reloaded_from": str(adapter_dir),
        }
        (adapter_dir / "smoke_inference.json").write_text(
            json.dumps(smoke_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    metadata = {
        "skill_id": skill_id,
        "version": version,
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "dataset": {
            "counts": split_counts,
            "sha256": {split: sha256(dataset_dir / f"{split}.jsonl") for split in REQUIRED_SPLITS},
        },
        "method": "4-bit QLoRA",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": sorted(lora.target_modules)},
        "training": {
            "epochs": epochs,
            "max_steps": max_steps,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "learning_rate": 1e-4,
            "min_image_pixels": MIN_IMAGE_PIXELS,
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "seed": 42,
        },
        "train_metrics": result.metrics,
        "smoke_inference": smoke_result,
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


def _generate(model, processor, row, dataset_dir, image_class, torch):
    images = []
    try:
        content = []
        for relative in row["images"]:
            image = image_class.open(dataset_dir / relative).convert("RGB")
            images.append(image)
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": row["prompt"]})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": row.get("system", "")}]},
            {"role": "user", "content": content},
        ]
        started = time.monotonic()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        model.eval()
        model.config.use_cache = True
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        text = processor.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        try:
            parsed = json.loads(text)
            valid_json = isinstance(parsed, dict)
        except json.JSONDecodeError:
            valid_json = False
        return {"output": text, "valid_json_object": valid_json, "latency_ms": elapsed_ms}
    finally:
        for image in images:
            image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapters"))
    parser.add_argument("--skill", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke-inference", action="store_true")
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
        epochs=args.epochs,
        max_steps=args.max_steps,
        smoke_inference=args.smoke_inference,
    )
    print(path)


if __name__ == "__main__":
    main()
