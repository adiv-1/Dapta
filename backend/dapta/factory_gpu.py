from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


def progress(**values: Any) -> None:
    print(json.dumps({"type": "progress", **values}, separators=(",", ":")), flush=True)


def restore_frozen_compute_dtype(model: Any, torch: Any) -> list[str]:
    """Keep frozen Qwen compute layers compatible with bf16 activations.

    PEFT intentionally promotes non-quantized parameters to float32 in
    ``prepare_model_for_kbit_training``. Qwen3-VL's untied output head (and,
    depending on the Transformers version, frozen vision projections) then
    receive bf16 activations outside an autocast-safe boundary. Restore only
    frozen compute layers to bf16; normalization layers remain float32.
    """
    converted: list[str] = []
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None:
        visual.requires_grad_(False)
        visual.to(dtype=torch.bfloat16)
        converted.append("model.visual")

    language_model = getattr(getattr(model, "model", None), "language_model", None)
    embedding = getattr(language_model, "embed_tokens", None)
    if embedding is not None:
        embedding.requires_grad_(False)
        embedding.to(dtype=torch.bfloat16)
        converted.append("model.language_model.embed_tokens")

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if (
            isinstance(module, torch.nn.Linear)
            and weight is not None
            and weight.dtype == torch.float32
            and not weight.requires_grad
        ):
            module.to(dtype=torch.bfloat16)
            converted.append(name)
    return sorted(set(converted))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    bundle_path = Path(args.bundle)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    result_path = bundle_path.parent / "result.json"
    result = evaluate_saved_adapter(bundle) if args.evaluate_only else run(bundle)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    progress(
        stage="complete",
        message=(
            "Saved adapter benchmark complete"
            if args.evaluate_only
            else "Adapter training and evaluation complete"
        ),
        percent=100,
    )


def evaluate_saved_adapter(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Rerun only the frozen benchmark for an existing PEFT adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    base_model = str(bundle["base_model"])
    adapter_dir = Path(str(bundle["adapter_dir"]))
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"saved adapter weights are missing: {adapter_dir}")
    test_cases = list(bundle["dataset"]["test"])[: int(bundle["max_test_examples"])]
    skill = _skill_prompt(bundle["skill_contract"])
    schema = bundle["skill_contract"].get("output_schema", {})
    progress(stage="evaluation", message="Loading base model for benchmark-only repair", percent=2)
    processor = AutoProcessor.from_pretrained(base_model)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa"),
    )
    model.eval()
    base_no_skill = evaluate_condition(
        model, processor, test_cases,
        "Complete this workflow and return valid JSON only: " + str(bundle["workflow"]),
        schema, condition="base_no_skill", progress_start=3, progress_end=28,
    )
    base_with_skill = evaluate_condition(
        model, processor, test_cases, skill, schema,
        condition="base_with_skill", progress_start=28, progress_end=53,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    lora_short = evaluate_condition(
        model, processor, test_cases,
        "Complete the learned workflow described here: "
        + str(bundle["workflow"]) + ". Return JSON only.",
        schema, condition="lora_short_prompt", progress_start=53, progress_end=97,
    )
    return {
        "test_cases": len(test_cases),
        "conditions": {
            "base_no_skill": base_no_skill,
            "base_with_skill": base_with_skill,
            "lora_short_prompt": lora_short,
        },
        "evaluation_only": True,
        "adapter_dir": str(adapter_dir),
    }


def run(bundle: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    base_model = str(bundle["base_model"])
    output_dir = Path(str(bundle["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    skill = _skill_prompt(bundle["skill_contract"])
    dataset = bundle["dataset"]
    train_cases = list(dataset["train"])[: int(bundle["max_train_examples"])]
    dev_cases = list(dataset["dev"])
    test_cases = list(dataset["test"])[: int(bundle["max_test_examples"])]
    progress(stage="loading", message=f"Loading {base_model} in 4-bit", percent=2)
    processor = AutoProcessor.from_pretrained(base_model)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa"),
    )
    model.config.use_cache = True
    progress(stage="baseline", message="Evaluating Qwen base and Qwen + Skill", percent=8)
    base_no_skill = evaluate_condition(
        model, processor, test_cases,
        "Complete this workflow and return valid JSON only: " + str(bundle["workflow"]),
        bundle["skill_contract"].get("output_schema", {}),
        condition="base_no_skill",
        progress_start=3,
        progress_end=8,
    )
    base_with_skill = evaluate_condition(
        model, processor, test_cases, skill,
        bundle["skill_contract"].get("output_schema", {}),
        condition="base_with_skill",
        progress_start=8,
        progress_end=14,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    converted_modules = restore_frozen_compute_dtype(model, torch)
    progress(
        stage="preparing",
        message="Aligned frozen Qwen compute layers with bf16 training",
        percent=14,
        converted_modules=converted_modules,
    )
    train_dataset = Dataset.from_list([_sft_case(case, skill) for case in train_cases])
    dev_dataset = Dataset.from_list([_sft_case(case, skill) for case in dev_cases])
    epochs = float(bundle["epochs"])
    max_length = int(bundle["max_length"])
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=max_length,
        completion_only_loss=True,
        packing=False,
        bf16=True,
        fp16=False,
        tf32=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=20261017,
    )

    class ProgressCallback(TrainerCallback):
        started = time.perf_counter()

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            del args, control, kwargs
            current = int(state.global_step)
            total = max(1, int(state.max_steps))
            elapsed = time.perf_counter() - self.started
            eta = elapsed / current * (total - current) if current else None
            progress(
                stage="training",
                message=f"Training step {current}/{total}",
                current=current,
                total=total,
                percent=round(15 + current / total * 62, 1),
                eta_seconds=round(eta) if eta is not None else None,
                loss=(logs or {}).get("loss"),
                gpu_memory_used_gb=round(torch.cuda.memory_allocated() / 1024**3, 2),
                gpu_utilization_percent=_gpu_utilization(),
            )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor,
        peft_config=lora,
        callbacks=[ProgressCallback()],
    )
    progress(stage="training", message="QLoRA training started", percent=15)
    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    model = trainer.model
    model.eval()
    model.config.use_cache = True
    progress(stage="evaluation", message="Evaluating trained LoRA on frozen TEST", percent=82)
    lora_short = evaluate_condition(
        model,
        processor,
        test_cases,
        "Complete the learned workflow described here: "
        + str(bundle["workflow"])
        + ". Return JSON only.",
        bundle["skill_contract"].get("output_schema", {}),
        condition="lora_short_prompt",
        progress_start=82,
        progress_end=97,
    )
    version = datetime_version()
    evaluation = {
        "test_cases": len(test_cases),
        "conditions": {
            "base_no_skill": base_no_skill,
            "base_with_skill": base_with_skill,
            "lora_short_prompt": lora_short,
        },
    }
    return {
        "version": version,
        "base_model": base_model,
        "adapter_dir": str(output_dir),
        "training_seconds": training_seconds,
        "train_examples": len(train_cases),
        "validation_examples": len(dev_cases),
        "test_examples": len(test_cases),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "trainer_metrics": _json_safe(train_result.metrics),
        "evaluation": evaluation,
    }


def evaluate_condition(
    model: Any,
    processor: Any,
    cases: list[Mapping[str, Any]],
    system_prompt: str,
    schema: Mapping[str, Any],
    *,
    condition: str,
    progress_start: float,
    progress_end: float,
) -> dict[str, Any]:
    import torch

    predictions: list[Any] = []
    elapsed: list[float] = []
    input_tokens = 0
    output_tokens = 0
    truncated_cases: list[str] = []
    max_new_tokens = int(os.environ.get("DAPTA_FACTORY_MAX_NEW_TOKENS", "4096"))
    for index, case in enumerate(cases):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _content(case["input"])},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")
        started = time.perf_counter()
        # PEFT keeps LoRA projection weights in float32 for stable updates.
        # Generation happens outside Trainer's mixed-precision context, so
        # restore autocast here when bf16 Qwen activations reach those layers.
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed.append(time.perf_counter() - started)
        generated = ids[:, inputs.input_ids.shape[1]:]
        output = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        predictions.append(_parse_json(output))
        input_tokens += int(inputs.input_ids.shape[1])
        output_tokens += int(generated.shape[1])
        if int(generated.shape[1]) >= max_new_tokens:
            truncated_cases.append(str(case.get("id") or index))
        progress(
            stage="evaluation",
            message=f"{condition}: {index + 1}/{len(cases)}",
            current=index + 1,
            total=len(cases),
            percent=round(
                progress_start
                + (index + 1) / max(1, len(cases)) * (progress_end - progress_start),
                1,
            ),
            gpu_utilization_percent=_gpu_utilization(),
        )
    result = score(cases, predictions, schema, elapsed, input_tokens, output_tokens)
    result["generation_max_new_tokens"] = max_new_tokens
    result["truncated_cases"] = truncated_cases
    return result


def score(
    cases: list[Mapping[str, Any]],
    predictions: list[Any],
    schema: Mapping[str, Any],
    elapsed: list[float],
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    exact = 0
    valid = 0
    true_positive = 0
    predicted_count = 0
    expected_count = 0
    failures: list[dict[str, Any]] = []
    for case, prediction in zip(cases, predictions):
        expected = case["output"]
        if _normalized(prediction) == _normalized(expected):
            exact += 1
        expected_fields = set(_flatten(expected))
        predicted_fields = set(_flatten(prediction))
        true_positive += len(expected_fields & predicted_fields)
        expected_count += len(expected_fields)
        predicted_count += len(predicted_fields)
        schema_ok = _schema_valid(prediction, schema)
        valid += int(schema_ok)
        if len(failures) < 10 and prediction != expected:
            failures.append({"case_id": case.get("id"), "schema_valid": schema_ok})
    precision = true_positive / predicted_count if predicted_count else 0
    recall = true_positive / expected_count if expected_count else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    total = max(1, len(cases))
    return {
        "field_f1": f1,
        "exact_form_rate": exact / total,
        "schema_pass_rate": valid / total,
        "mean_latency_seconds": sum(elapsed) / len(elapsed) if elapsed else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "failures": failures,
    }


def _sft_case(case: Mapping[str, Any], skill: str) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": skill}]},
            {"role": "user", "content": [{"type": "text", "text": _content(case["input"])}]},
        ],
        "completion": [
            {"role": "assistant", "content": [{
                "type": "text",
                "text": json.dumps(case["output"], ensure_ascii=False, separators=(",", ":")),
            }]},
        ],
    }


def _skill_prompt(contract: Mapping[str, Any]) -> str:
    return (
        "Follow this workflow contract exactly and return JSON only.\n"
        + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
    )


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _gpu_utilization() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip().splitlines()[0]
        return float(output)
    except Exception:
        return None


def _parse_json(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = min((position for position in (value.find("{"), value.find("[")) if position >= 0), default=-1)
        if start >= 0:
            for end in range(len(value), start, -1):
                try:
                    return json.loads(value[start:end])
                except json.JSONDecodeError:
                    continue
        return {"_invalid_json": True}


def _flatten(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_flatten(item, f"{prefix}[{index}]"))
        return result
    return [f"{prefix}={json.dumps(value, sort_keys=True, ensure_ascii=False)}"]


def _schema_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    if not schema:
        return isinstance(value, (dict, list)) and "_invalid_json" not in value
    try:
        import jsonschema
        jsonschema.validate(value, schema)
        return True
    except Exception:
        return False


def _normalized(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            result[key] = None
        else:
            result[key] = item
    return result


def datetime_version() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")


if __name__ == "__main__":
    main()
