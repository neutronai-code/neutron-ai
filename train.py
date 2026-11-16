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
import argparse, inspect, json, os, re, sys, time
from repair import CODE_SYS, system_prompt_for
from sandbox import grpo_reward_func, reasoning_format_reward_func


def is_vision_model(model_name: str) -> bool:
    """Qwen3.5 and similar ship as image-text-to-text models, which need Unsloth's FastModel loader."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return False
    archs = getattr(cfg, "architectures", None) or []
    return hasattr(cfg, "vision_config") or any(a.endswith("ForConditionalGeneration") for a in archs)


def is_qwen35(model_name: str) -> bool:
    """Qwen3.5 (Gated DeltaNet hybrid). Unsloth: 4-bit QLoRA hurts these models more than usual, so train in bf16."""
    return bool(re.search(r"qwen3[._]5", model_name, re.I))


def model_billions(model_name: str) -> float:
    """Parameter count from the name, e.g. 'Qwen/Qwen3.5-4B' -> 4.0 (0 if the name doesn't say)."""
    m = re.search(r"(\d+(?:\.\d+)?)B(?![a-z])", model_name.split("/")[-1], re.I)
    return float(m.group(1)) if m else 0.0


def hf_preflight(repo_id: str):
    """Create (or confirm write access to) the HF repo BEFORE training, so a bad name or token fails in minute one
    instead of after hours of paid GPU time."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        sys.exit(f"[NeutronAI] Can't write to Hugging Face repo '{repo_id}': {e}\n"
                 f"Check HF_TOKEN has write access and the namespace is yours (e.g. neutronai-code/...).")
    print(f"[NeutronAI] HF upload target OK: {repo_id} (logged in as {user})", flush=True)


def make_prompt_renderer(tokenizer, thinking: str, max_completion: int):
    """Return (render_fn, thinking_on). render_fn is None to keep TRL's conversational prompts (old behaviour).

    Models whose chat template has a thinking switch (Qwen3/3.5) think by default and need 32K+ tokens for it;
    with a small completion budget nearly every answer would be cut off mid-thought and score 0. So:
    `auto` turns thinking on only when the completion budget is >= 8192 tokens (e.g. A100 runs), else off."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" not in template:
        return None, False
    on = thinking == "on" or (thinking == "auto" and max_completion >= 8192)

    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=on)

    return render, on


def _curriculum_key(r: dict) -> float:
    """Most informative first: measured ~50% pass rate, then unmeasured tasks (assumed hard), then never-solved."""
    rate = r.get("baseline_pass_rate")
    return 0.3 if rate is None else abs(rate - 0.5)


def load_rl_dataset(source: str, render=None, tokenizer=None, max_prompt_tokens: int = 0):
    from datasets import Dataset, load_dataset

    fmt = render or (lambda messages: messages)
    if source == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        return Dataset.from_list([
            {
                "prompt": fmt([
                    {"role": "system", "content": CODE_SYS},
                    {"role": "user", "content": f"{r['prompt']}\n\nExample check:\n{r['test_list'][0]}"},
                ]),
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
    rows.sort(key=_curriculum_key)  # stable: ties keep file order
    if tokenizer is not None and max_prompt_tokens and render is not None:
        # TRL may ignore max_prompt_length; drop overlong prompts so a long repo task can't OOM the GPU.
        def n_tokens(r):
            text = render([{"role": "system", "content": system_prompt_for(r)}, {"role": "user", "content": r["prompt"]}])
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        before = len(rows)
        rows = [r for r in rows if n_tokens(r) <= max_prompt_tokens]
        if len(rows) < before:
            print(f"[NeutronAI] dropped {before - len(rows)} tasks with prompts over {max_prompt_tokens} tokens", flush=True)
    return Dataset.from_list([
        {
            "prompt": fmt([
                {"role": "system", "content": system_prompt_for(r)},
                {"role": "user", "content": r["prompt"]},
            ]),
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


def _make_hub_checkpoint_callback(repo_id: str):
    """After every checkpoint save, upload it to `<repo>/last-checkpoint` in the background, so a crashed or
    timed-out VM loses at most one save interval. `--resume` picks it back up on a fresh VM."""
    from huggingface_hub import HfApi
    from transformers import TrainerCallback

    api = HfApi()

    class HubCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt):
                print(f"[NeutronAI] uploading checkpoint-{state.global_step} -> {repo_id}/last-checkpoint", flush=True)
                api.upload_folder(repo_id=repo_id, folder_path=ckpt, path_in_repo="last-checkpoint",
                                  commit_message=f"checkpoint step {state.global_step}", run_as_future=True)
            return control

    return HubCheckpointCallback()


def _find_resume_checkpoint(out_dir: str, repo_id: str) -> str | None:
    """Latest local checkpoint in out_dir, else the hub's last-checkpoint (downloaded), else None."""
    if os.path.isdir(out_dir):
        ckpts = sorted((d for d in os.listdir(out_dir) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
        if ckpts:
            return os.path.join(out_dir, ckpts[-1])
    if repo_id:
        from huggingface_hub import snapshot_download

        try:
            local = snapshot_download(repo_id, allow_patterns=["last-checkpoint/*"], local_dir=f"{out_dir}-resume")
        except Exception as e:
            print(f"[NeutronAI] no hub checkpoint to resume from ({e})", flush=True)
            return None
        path = os.path.join(local, "last-checkpoint")
        if os.path.isfile(os.path.join(path, "trainer_state.json")):
            print(f"[NeutronAI] resuming from {repo_id}/last-checkpoint", flush=True)
            return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "dpo", "grpo", "fim-grpo"], default="grpo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default="tasks.jsonl")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Hard wall-clock stop to protect Vultr credits (0=off)")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from the latest checkpoint (local, else <--push-hf>/last-checkpoint on the Hub)")
    ap.add_argument("--save-steps", type=int, default=0, help="Checkpoint every N steps (0 = auto, at most 50)")
    ap.add_argument("--max-prompt-tokens", type=int, default=1024, help="Drop RL tasks whose prompt is longer")
    ap.add_argument("--export-gguf", action="store_true")
    ap.add_argument("--push-hf", default="")
    ap.add_argument("--out", default="hadron-v0.9")
    ap.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                    help="Qwen3/3.5 thinking mode. auto = on only if the completion budget (max-seq-len - 1024) >= 8192")
    ap.add_argument("--load-in-4bit", choices=["auto", "yes", "no"], default="auto",
                    help="auto = 4-bit below 40GB VRAM, bf16 on bigger GPUs (A100 80GB)")
    args = ap.parse_args()
    if args.push_hf:
        hf_preflight(args.push_hf)

    # ponytail: single-GPU Unsloth handles 1.5B-32B; upgrade to multi-node verl/OpenRLHF when funded past 1x H100.
    import torch
    from unsloth import FastLanguageModel, FastModel, PatchDPOTrainer, is_bfloat16_supported

    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
    # Use vLLM fast_inference on >=24GB single GPUs; on 16GB cards (e.g. A16/T4) use Unsloth native generation to avoid OOM!
    use_vllm = (args.mode in ("grpo", "fim-grpo")) and (vram_gb >= 22.0)
    if vram_gb > 0 and vram_gb < 22.0 and args.num_generations > 4:
        args.num_generations = 4
    load_4bit = args.load_in_4bit == "yes" or (args.load_in_4bit == "auto" and vram_gb < 40.0)
    if is_qwen35(args.model) and args.load_in_4bit != "yes":
        # Qwen3.5 must train in bf16. Refuse early if the weights clearly won't fit, instead of silently degrading.
        load_4bit = False
        need_gb = model_billions(args.model) * 2.3 + 4.0  # bf16 weights (+vision tower) + LoRA/activation headroom
        if vram_gb and need_gb > vram_gb:
            sys.exit(f"[NeutronAI] {args.model} needs ~{need_gb:.0f}GB VRAM in bf16 but this GPU has {vram_gb:.0f}GB. "
                     f"Unsloth advises against 4-bit training for Qwen3.5. Use a smaller model (Qwen/Qwen3.5-4B), "
                     f"a bigger GPU, or force it with --load-in-4bit yes.")
    elif is_qwen35(args.model) and load_4bit:
        print("[NeutronAI] WARNING: Unsloth advises against 4-bit training on Qwen3.5 (quality loss).", flush=True)
    vision = is_vision_model(args.model)
    print(f"[NeutronAI] model={args.model} vision_arch={vision} vram={vram_gb:.0f}GB 4bit={load_4bit} vllm={use_vllm}", flush=True)

    loader = FastModel if vision else FastLanguageModel
    load_kwargs = dict(model_name=args.model, max_seq_length=args.max_seq_len, load_in_4bit=load_4bit)
    if use_vllm or not vision:  # keep the original FastLanguageModel call unchanged for text models
        load_kwargs.update(fast_inference=use_vllm, gpu_memory_utilization=0.65 if vram_gb >= 22.0 else 0.55)
    model, tokenizer = loader.from_pretrained(**load_kwargs)
    if not hasattr(model, "peft_config"):
        if vision:
            # Text-only training: LoRA on the language model's attention + MLP, vision tower frozen.
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=32,
                lora_alpha=32,
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        else:
            model = FastLanguageModel.get_peft_model(
                model,
                r=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
    # Vision models come with a multimodal processor; all our data is text, so use its inner text tokenizer.
    if vision:
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    render, thinking_on = make_prompt_renderer(tokenizer, args.thinking, args.max_seq_len - 1024)
    print(f"[NeutronAI] thinking={'on' if thinking_on else 'off'} (max completion {args.max_seq_len - 1024} tokens)", flush=True)

    callbacks = [_make_wallclock_guard(args.max_minutes)] if args.max_minutes > 0 else []
    if args.push_hf:
        callbacks.append(_make_hub_checkpoint_callback(args.push_hf))
    save_steps = args.save_steps or min(50, max(10, args.steps // 4))

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
                save_steps=save_steps,
                save_total_limit=2,
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
                save_steps=save_steps,
                save_total_limit=2,
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
            # TRL requires the generation batch (per_device x grad_accum) to be divisible by num_generations;
            # a fixed 4 broke the >=22GB path where num_generations stays at 6. Unchanged (4) on 16GB cards.
            gradient_accumulation_steps=args.num_generations,
            num_generations=args.num_generations,
            max_prompt_length=1024,
            max_completion_length=args.max_seq_len - 1024,
            max_steps=args.steps,
            logging_steps=1,
            save_steps=save_steps,
            save_total_limit=2,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            report_to="none",
        )
        grpo_params = inspect.signature(GRPOConfig.__init__).parameters
        if "loss_type" in grpo_params:
            cfg_kwargs["loss_type"] = "dr_grpo"
        if "shuffle_dataset" in grpo_params:
            cfg_kwargs["shuffle_dataset"] = False  # keep the curriculum order (most informative tasks first)

        if args.mode == "fim-grpo":
            from datasets import Dataset
            from fim import fim_grpo_reward_func

            with open(args.data, encoding="utf-8") as f:
                fim_rows = [json.loads(l) for l in f if l.strip()]
            fmt = render or (lambda messages: messages)
            fim_ds = Dataset.from_list([
                {
                    "prompt": fmt([
                        {"role": "system", "content": f"Complete ONLY the missing `{r.get('lang', 'python')}` code between <PREFIX> and <SUFFIX> inside a fenced block."},
                        {"role": "user", "content": r.get("fim_prompt") or f"<PREFIX>\n{r['prefix']}</PREFIX>\n<SUFFIX>\n{r['suffix']}</SUFFIX>"},
                    ]),
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
                train_dataset=load_rl_dataset(args.data, render, tokenizer, args.max_prompt_tokens),
            )

    resume_from = _find_resume_checkpoint(args.out, args.push_hf) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    if args.push_hf:
        try:
            model.push_to_hub_merged(args.push_hf, tokenizer, save_method="merged_16bit")
        except Exception as e:  # never lose a finished run: at least upload the LoRA adapter
            print(f"[NeutronAI] Merged upload failed ({e}); uploading the LoRA adapter instead.", flush=True)
            from huggingface_hub import HfApi

            HfApi().upload_folder(repo_id=args.push_hf, folder_path=args.out, path_in_repo="lora_adapter")
        if args.export_gguf:
            try:
                model.push_to_hub_gguf(f"{args.push_hf}-GGUF", tokenizer, quantization_method="q4_k_m")
            except Exception as e:
                print(f"[NeutronAI] GGUF export/upload failed ({e}); the bf16 model is already uploaded.", flush=True)
    elif args.export_gguf:
        model.save_pretrained_gguf(f"{args.out}-gguf", tokenizer, quantization_method="q4_k_m")
    print(f"Finished {args.mode.upper()} -> {args.out}")


if __name__ == "__main__":
    main()
