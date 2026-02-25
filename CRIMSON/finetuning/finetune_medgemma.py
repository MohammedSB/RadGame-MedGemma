"""
Fine-tune MedGemma-4B-IT to produce CRIMSON evaluation scores.

Input:  CRIMSON prompt (excl. CONTEXT_GUIDELINES) with GT + candidate reports
Output: raw_evaluation JSON

Supports multi-GPU via Accelerate / FSDP and uses LoRA (r=16) on the
language model with bf16 mixed precision.

Usage (single node, launched via accelerate):
    accelerate launch finetune_medgemma.py \
        --train_jsonl ../data/finetuned_medgemma/train_data.jsonl \
        --output_dir ../data/finetuned_medgemma/checkpoints \
        --model_id google/medgemma-4b-it \
        --num_samples 300
"""

import argparse
import json
import math
import os
import random
import traceback

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dataset import CRIMSONDataset, collate_fn


# ------------------------------------------------------------------ helpers
def print_main(msg: str):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        print(msg)


def load_jsonl(
    path: str,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load a JSONL file, optionally shuffling and limiting the number of samples.

    With shuffle=True, all valid entries are loaded, shuffled with a fixed seed,
    then truncated to max_samples.  This avoids bias from sequential ordering.
    """
    data = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            # Skip entries with no raw_evaluation
            if entry.get("raw_evaluation") is not None:
                data.append(entry)
            # Early exit when not shuffling
            if not shuffle and max_samples is not None and len(data) >= max_samples:
                break

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)

    if max_samples is not None:
        data = data[:max_samples]

    return data


# ------------------------------------------------------------------ callback
class SampleGenerationCallback(TrainerCallback):
    """Generate and print model responses on a few held-out prompts
    every `generate_every` training steps so we can eyeball quality."""

    def __init__(self, dataset, tokenizer, generate_every: int = 25,
                 num_samples: int = 3, max_new_tokens: int = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.generate_every = generate_every
        self.max_new_tokens = max_new_tokens

        # Pick fixed sample indices (spread across the dataset)
        n = len(dataset)
        step = max(n // num_samples, 1)
        self.sample_indices = [i * step for i in range(num_samples) if i * step < n]

        # Cache the raw prompts and ground-truth targets for display
        self.prompts = []       # raw prompt text (user message)
        self.gt_targets = []    # ground-truth target strings (full)
        for idx in self.sample_indices:
            sample = dataset.samples[idx]
            self.prompts.append(sample["prompt"])
            self.gt_targets.append(sample["target"])

    @staticmethod
    def _unwrap_model(model):
        """Unwrap DDP / FSDP / PeftModel wrappers to get the base model
        for generation, so we don't break DDP synchronization."""
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        return unwrapped

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.generate_every != 0:
            return
        # Only run on rank 0
        if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) != 0:
            return
        if model is None:
            return

        # IMPORTANT: unwrap from DDP before calling generate, otherwise
        # only rank 0 would call DDP.forward and break gradient sync.
        unwrapped = self._unwrap_model(model)
        unwrapped.eval()
        device = next(unwrapped.parameters()).device

        print(f"\n{'='*80}")
        print(f"  Sample generations at step {state.global_step}")
        print(f"{'='*80}")

        for i, (prompt, gt) in enumerate(zip(self.prompts, self.gt_targets)):
            messages = [
                {"role": "user", "content": prompt},
            ]
            enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            # token_type_ids: all 0 for text-only
            token_type_ids = torch.zeros_like(input_ids)

            with torch.no_grad():
                try:
                    out = unwrapped.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                    generated = self.tokenizer.decode(
                        out[0][input_ids.shape[1]:], skip_special_tokens=True
                    )
                except Exception as e:
                    generated = f"[generation failed: {e}]"

            print(f"\n{'─'*80}")
            print(f"--- Sample {i+1} (idx={self.sample_indices[i]}) ---")
            print(f"{'─'*80}")
            print(f"\n[INPUT PROMPT]\n{prompt}")
            print(f"\n[GROUND TRUTH]\n{gt}")
            print(f"\n[MODEL PREDICTION]\n{generated}")

        print(f"\n{'='*80}\n")
        unwrapped.train()


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma-4B for CRIMSON evaluation scoring"
    )

    # Data
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to train_data.jsonl",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to use from the JSONL (-1 = all, default: -1)",
    )

    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/medgemma-4b-it",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for model weights",
    )

    # Training hyper-parameters
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4500)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["epoch", "steps", "no"],
                        help="When to save checkpoints (default: epoch)")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save every N steps (only used if save_strategy=steps)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max checkpoints to keep on disk (default: 3)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                        help="Num dataloader workers per GPU (default: 2)")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    # ---- reproducibility ----
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- load data ----
    max_samples = None if args.num_samples < 0 else args.num_samples
    print_main(f"Loading data from {args.train_jsonl} "
               f"({'all' if max_samples is None else max_samples} samples, shuffled) ...")
    data = load_jsonl(args.train_jsonl, max_samples=max_samples, shuffle=True, seed=args.seed)
    print_main(f"  Loaded {len(data)} valid samples")

    # ---- load processor (tokenizer + image processor) ----
    print_main(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for causal LM (matches our collate_fn)
    tokenizer.padding_side = "left"

    # ---- build dataset ----
    print_main("Building dataset ...")
    dataset = CRIMSONDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print_main(f"  Dataset size: {len(dataset)}")

    # Print a token-length summary for the first few samples
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        lengths = [dataset[i]["input_ids"].shape[0] for i in range(min(10, len(dataset)))]
        print(f"  Token lengths (first {len(lengths)} samples): "
              f"min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

        # ---- diagnostic: verify label masking on first sample ----
        diag = dataset[0]
        n_masked = (diag["labels"] == -100).sum().item()
        n_target = (diag["labels"] != -100).sum().item()
        target_ids = diag["labels"][diag["labels"] != -100]
        print(f"  [DIAG] Sample 0: {n_masked} prompt tokens masked, "
              f"{n_target} target tokens visible ({n_target/(n_masked+n_target):.1%})")
        print(f"  [DIAG] Target preview: "
              f"{tokenizer.decode(target_ids[:40], skip_special_tokens=True)[:200]}...")
        print(f"  [DIAG] Last target token is EOS: "
              f"{target_ids[-1].item() == tokenizer.eos_token_id}")

    # ---- load model ----
    # MedGemma is an Image-Text-to-Text model (based on Gemma 3)
    # so it must be loaded with AutoModelForImageTextToText even for text-only use
    print_main(f"Loading model: {args.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        # Don't set device_map when using accelerate multi-GPU
        device_map=None,
    )

    # ---- apply LoRA ----
    print_main("Applying LoRA ...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        model.print_trainable_parameters()

    # ---- training arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        # gradient checkpointing saves memory at the cost of ~20% slower training
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # DDP: MedGemma has vision params unused in text-only LoRA finetuning
        ddp_find_unused_parameters=True,
    )

    # ---- sample generation callback ----
    num_gpus = max(1, torch.cuda.device_count())
    samples_per_gpu = math.ceil(len(dataset) / num_gpus)
    microbatches_per_epoch = math.ceil(samples_per_gpu / args.batch_size)
    steps_per_epoch = max(1, microbatches_per_epoch // args.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    gen_every = max(1, total_steps // 4)  # ~4 sample prints per run
    print_main(f"  Total estimated steps: {total_steps}, will generate samples every {gen_every} steps")

    sample_cb = SampleGenerationCallback(
        dataset=dataset,
        tokenizer=tokenizer,
        generate_every=gen_every,
        num_samples=3,       # 3 fixed samples
        max_new_tokens=4096, # full CRIMSON output can be long
    )

    # ---- trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[sample_cb],
    )

    # ---- train ----
    print_main(f"Starting training for {args.num_epochs} epochs ...")
    print_main(f"  Effective batch size: "
               f"{args.batch_size * args.gradient_accumulation_steps * num_gpus} "
               f"(per_device={args.batch_size} x grad_accum={args.gradient_accumulation_steps} "
               f"x gpus={num_gpus})")
    print_main(f"  Accelerate distributed type: {trainer.args.parallel_mode}")
    print_main(f"  World size (num processes): {trainer.args.world_size}")
    print_main(f"  Local rank: {trainer.args.local_process_index}, "
               f"Global rank: {trainer.args.process_index}")
    try:
        trainer.train()
    except Exception:
        traceback.print_exc()
        raise

    # ---- save ----
    final_dir = os.path.join(args.output_dir, "final")
    print_main(f"Saving final model to {final_dir} ...")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print_main("Done!")


if __name__ == "__main__":
    main()
