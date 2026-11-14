"""NeutronAI v0.9 — Four-Mode Post-Training Pipeline for Hadron & Quark (SFT -> DPO -> Dr.GRPO -> Repo-FIM-GRPO).

Full NeutronAI training stack:
  Stage 1 (`--mode sft`)      : Rejection-sampled verified `<think> + code/patch` warmup (`sft_verified.jsonl`).
  Stage 2 (`--mode dpo`)      : Contrastive preference tuning on `dpo_repairs.jsonl` harvested by `repair.py`
                                (teaches Hadron to prefer passing fixes over its own exact Turn-1 bugs).
  Stage 3 (`--mode grpo`)     : Execution-verified RLVR for **Hadron-7B / 32B** (`tasks.jsonl`, `rl_curriculum.jsonl`, `mbpp`)
                                with dense per-assert + speed rewards, conciseness shaping, `beta=0.0`, and `dr_grpo` loss.
  Stage 4 (`--mode fim-grpo`) : Execution-verified Cross-File `<|file_sep|>` Repo-FIM RLVR for **Quark-1.5B** (`fim_tasks.jsonl`).

Vultr Cost-Guard:
  Pass `--max-minutes 45` so a runaway hangout never drains your $300 GPU credits, and DESTROY the Vultr VM right after `--push-hf`!
"""
import argparse, inspect, json, time
from repair import CODE_SYS, system_prompt_for
from sandbox import grpo_reward_func, reasoning_format_reward_func


def load_rl_dataset(source: str):
    from datasets import Dataset, load_dataset

    if source == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        return Dataset.from_list([
            {
                "prompt": [
                    {"role": "system", "content": CODE_SYS},
                    {"role": "user", "content": f"{r['prompt']}\n\nExample check:\n{r['test_list'][0]}"},
                ],
                "test": "\n".join(r["test_imports"] + r["test_list"]),
                "starter_code": "",
                "files": "",
                "lang": "python",
                "perf_budget_ms": 0.0,
            }
            for r in ds
        ])

    with open(source, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: abs(r.get("baseline_pass_rate", 0.5) - 0.5))
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": system_prompt_for(r)},
                {"role": "user", "content": r["prompt"]},
            ],
            "test": r["test"],
            "starter_code": r.get("starter_code", ""),
            "files": json.dumps(r["files"]) if r.get("files") else "",
            "lang": r.get("lang", "python"),
            "perf_budget_ms": float(r.get("perf_budget_ms", 0.0)),
        }
        for r in rows
    ])


def load_sft_dataset(path: str, tokenizer):
    from datasets import Dataset

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list([
        {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)}
        for r in rows
    ])


def load_dpo_dataset(path: str, tokenizer):
    from datasets import Dataset

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list([
        {
            "prompt": tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True),
            "chosen": r["chosen"][0]["content"],
            "rejected": r["rejected"][0]["content"],
        }
        for r in rows
    ])


def _make_wallclock_guard(max_minutes: float):
    from transformers import TrainerCallback

    deadline = time.monotonic() + max_minutes * 60.0

    class BudgetGuardCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() >= deadline:
                print(f"\n[BudgetGuard] Reached {max_minutes} min wall-clock cap at step {state.global_step}. Saving cleanly...")
                control.should_save = True
                control.should_training_stop = True
            return control

    return BudgetGuardCallback()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "dpo", "grpo", "fim-grpo"], default="grpo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Hard wall-clock stop to protect Vultr credits (0=off)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export-gguf", action="store_true")
    ap.add_argument("--push-hf", default="")
    ap.add_argument("--out", default="hadron-v0.9")
    args = ap.parse_args()

    # ponytail: single-GPU Unsloth handles 1.5B-32B; upgrade to multi-node verl/OpenRLHF when funded past 1x H100.
    from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        fast_inference=(args.mode in ("grpo", "fim-grpo")),
        gpu_memory_utilization=0.65,
    )
    if not hasattr(model, "peft_config"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    callbacks = [_make_wallclock_guard(args.max_minutes)] if args.max_minutes > 0 else []

    if args.mode == "sft":
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=load_sft_dataset(args.data, tokenizer),
            callbacks=callbacks,
            args=SFTConfig(
                output_dir=args.out,
                dataset_text_field="text",
                max_seq_length=args.max_seq_len,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                max_steps=args.steps,
                logging_steps=1,
                save_steps=max(10, args.steps // 3),
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                report_to="none",
            ),
        )
    elif args.mode == "dpo":
        PatchDPOTrainer()
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            processing_class=tokenizer,
            train_dataset=load_dpo_dataset(args.data, tokenizer),
            callbacks=callbacks,
            args=DPOConfig(
                output_dir=args.out,
                beta=0.1,
                max_length=args.max_seq_len,
                max_prompt_length=1024,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=5e-6,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                max_steps=args.steps,
                logging_steps=1,
                save_steps=max(10, args.steps // 3),
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                report_to="none",
            ),
        )
    else:
        from trl import GRPOConfig, GRPOTrainer

        cfg_kwargs = dict(
            output_dir=args.out,
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            beta=0.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=args.num_generations,
            max_prompt_length=1024,
            max_completion_length=args.max_seq_len - 1024,
            max_steps=args.steps,
            logging_steps=1,
            save_steps=max(10, args.steps // 4),
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to="none",
        )
        if "loss_type" in inspect.signature(GRPOConfig.__init__).parameters:
            cfg_kwargs["loss_type"] = "dr_grpo"

        if args.mode == "fim-grpo":
            from datasets import Dataset
            from fim import fim_grpo_reward_func

            with open(args.data, encoding="utf-8") as f:
                fim_rows = [json.loads(l) for l in f if l.strip()]
            fim_ds = Dataset.from_list([
                {
                    "prompt": [
                        {"role": "system", "content": f"Complete ONLY the missing `{r.get('lang', 'python')}` code between <PREFIX> and <SUFFIX> inside a fenced block."},
                        {"role": "user", "content": r.get("fim_prompt") or f"<PREFIX>\n{r['prefix']}</PREFIX>\n<SUFFIX>\n{r['suffix']}</SUFFIX>"},
                    ],
                    "prefix": r["prefix"],
                    "suffix": r["suffix"],
                    "test": r["test"],
                    "lang": r.get("lang", "python"),
                    "repo_files": json.dumps(r["repo_files"]) if r.get("repo_files") else "",
                    "target_path": r.get("target_path", ""),
                }
                for r in fim_rows
            ])
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[fim_grpo_reward_func],
                callbacks=callbacks,
                args=GRPOConfig(**cfg_kwargs),
                train_dataset=fim_ds,
            )
        else:
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[grpo_reward_func, reasoning_format_reward_func],
                callbacks=callbacks,
                args=GRPOConfig(**cfg_kwargs),
                train_dataset=load_rl_dataset(args.data),
            )

    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    if args.export_gguf:
        model.save_pretrained_gguf(f"{args.out}-gguf", tokenizer, quantization_method="q4_k_m")
    if args.push_hf:
        model.push_to_hub_merged(args.push_hf, tokenizer, save_method="merged_16bit")
    print(f"Finished {args.mode.upper()} -> {args.out}")


if __name__ == "__main__":
    main()
