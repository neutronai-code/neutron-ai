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
import argparse, inspect, json, os, random, re, sys, time
from repair import CODE_SYS, system_prompt_for, task_prompt
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
            text = render([{"role": "system", "content": system_prompt_for(r)}, {"role": "user", "content": task_prompt(r)}])
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        before = len(rows)
        rows = [r for r in rows if n_tokens(r) <= max_prompt_tokens]
        if len(rows) < before:
            print(f"[NeutronAI] dropped {before - len(rows)} tasks with prompts over {max_prompt_tokens} tokens", flush=True)
    return Dataset.from_list([
        {
            "prompt": fmt([
                {"role": "system", "content": system_prompt_for(r)},
                {"role": "user", "content": task_prompt(r)},
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


def load_fim_sft_dataset(path: str, tokenizer, max_len: int, style: str = "auto"):
    """FIM-SFT rows ({fim_prompt, oracle_middle}, or next-edit rows {prompt, target} used as they are) -> token ids
    with the loss only on the fill + <|endoftext|>,
    the token Quark stops on at inference. Rows with "loss": "all" (quark_data_v04.py's whole-file rows) put the
    loss on every token instead. Rows longer than max_len tokens are dropped.

    The rows are in Qwen's FIM format; a model with StarCoder-style tokens (JetBrains Mellum: `<fim_prefix>`,
    `<filename>`) gets them converted to its own suffix-first format (fim.to_mellum_prompt). Both end on <|endoftext|>."""
    from datasets import Dataset

    from fim import convert_fim_prompt

    vocab = tokenizer.get_vocab()
    if style == "auto":  # from the tokenizer; "qwen-spm" (Qwen's tokens, suffix first) has to be asked for
        style = "starcoder" if "<fim_prefix>" in vocab else "granite" if "<|filename|>" in vocab else "qwen"
    if style != "qwen":
        print(f"[NeutronAI] FIM-SFT: {style}-style FIM tokens: rows converted to that format", flush=True)
    eos = tokenizer.eos_token or "<|endoftext|>"  # Qwen / Mellum: <|endoftext|>; Granite: <|end_of_text|>
    out, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # Next-edit rows (nes_data.py: {"prompt", "target"}) are already in their final form; FIM rows get converted
            prompt = r["prompt"] if "prompt" in r else convert_fim_prompt(r["fim_prompt"], style)
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c = tokenizer(r.get("target", r.get("oracle_middle")) + eos, add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_len:
                dropped += 1
                continue
            out.append({"input_ids": p + c, "labels": (p if r.get("loss") == "all" else [-100] * len(p)) + c,
                        "length": len(p) + len(c)})  # for group_by_length batching
    print(f"[NeutronAI] FIM-SFT: {len(out)} examples ({dropped} longer than {max_len} tokens dropped)", flush=True)
    return Dataset.from_list(out)


def _fim_sft_collator(pad_id: int):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    return collate


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


def _make_memory_report():
    """Peak GPU memory after the first steps (the longest batch comes first with group_by_length, so a run that gets
    past them won't run out of memory later) and every 500 steps."""
    import torch
    from transformers import TrainerCallback

    class MemoryReport(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available() and (state.global_step <= 2 or state.global_step % 500 == 0):
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[NeutronAI] step {state.global_step}: peak GPU memory "
                      f"{torch.cuda.max_memory_allocated() / 1e9:.1f} of {total:.0f} GB", flush=True)
            return control

    return MemoryReport()


def _make_hub_checkpoint_callback(repo_id: str):
    """After every checkpoint save, upload it to `<repo>/last-checkpoint` in the background, so a crashed or
    timed-out VM loses at most one save interval. `--resume` picks it back up on a fresh VM."""
    from huggingface_hub import HfApi
    from transformers import TrainerCallback

    api = HfApi()

    class HubCheckpointCallback(TrainerCallback):
        pending = []  # the uploads still running: main() waits for them before it exits

        def on_save(self, args, state, control, **kwargs):
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt):
                print(f"[NeutronAI] uploading checkpoint-{state.global_step} -> {repo_id}/last-checkpoint", flush=True)
                self.pending.append(api.upload_folder(repo_id=repo_id, folder_path=ckpt, path_in_repo="last-checkpoint",
                                                      commit_message=f"checkpoint step {state.global_step}",
                                                      run_as_future=True))
            return control

    return HubCheckpointCallback()


def _find_resume_checkpoint(out_dir: str, repo_id: str) -> str | None:
    """Latest complete local checkpoint in out_dir, else the hub's last-checkpoint (downloaded), else None.
    A local checkpoint without weights (a save cut short, or a notebook disk that dropped the big files after a
    restart, as happened on molab) is skipped, and the hub copy is used instead."""
    def complete(path: str) -> bool:
        names = set(os.listdir(path))
        weights = {"adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin",
                   "model.safetensors.index.json", "pytorch_model.bin.index.json"}
        return "trainer_state.json" in names and bool(names & weights) and all(
            os.path.getsize(os.path.join(path, n)) > 0 for n in names & weights)

    if os.path.isdir(out_dir):
        ckpts = sorted((d for d in os.listdir(out_dir) if re.fullmatch(r"checkpoint-\d+", d)),
                       key=lambda d: int(d.split("-")[1]))
        for d in reversed(ckpts):
            if complete(os.path.join(out_dir, d)):
                return os.path.join(out_dir, d)
            print(f"[NeutronAI] skipping incomplete local {d}", flush=True)
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


def hf_model_type(model_name: str) -> str:
    """config.json's model_type, read from the Hub (or a local folder) without importing the model code."""
    try:
        if os.path.isdir(model_name):
            path = os.path.join(model_name, "config.json")
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_name, "config.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("model_type", "")
    except Exception:
        return ""


PLAIN_HF_TYPES = {"mellum"}  # architectures trained with plain transformers + PEFT instead of Unsloth


def _load_plain_bf16(model_name: str, weights: str | None = None):
    """bf16 model with plain transformers, gradient checkpointing on: model_name's tokenizer, and its weights or those
    in the folder `weights` (a training checkpoint). The experts of a mixture-of-experts model run on transformers'
    grouped-matmul kernel when this torch/GPU has it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src = weights or model_name
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    try:
        model = AutoModelForCausalLM.from_pretrained(src, experts_implementation="grouped_mm", **kw)
    except (TypeError, ValueError) as e:  # no MoE kernel option for this model / version: the default is fine
        print(f"[NeutronAI] experts_implementation=grouped_mm not available ({e}); using the default", flush=True)
        model = AutoModelForCausalLM.from_pretrained(src, **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def load_plain_full(model_name: str, resume_from: str | None = None):
    """bf16 model with every weight trainable, for --full-finetune on architectures in PLAIN_HF_TYPES. A
    mixture-of-experts model's router (Mellum2: mlp.gate) stays frozen, as in the LoRA runs, so which experts each
    token goes to doesn't drift (the training loss has no load-balancing term).

    Resuming loads the checkpoint's weights here, through from_pretrained. The Trainer's own resume reloads them with
    a plain load_state_dict, which doesn't undo transformers' checkpoint conversion: Mellum2's experts are saved one
    tensor per expert but live fused in memory, so they'd silently stay at their starting values. (The Trainer's
    reload then only sets the other weights again, to the same values.)"""
    model, tokenizer = _load_plain_bf16(model_name, weights=resume_from)
    if resume_from:
        print(f"[NeutronAI] weights loaded from {resume_from}", flush=True)
    frozen = [n for n, prm in model.named_parameters() if ".mlp.gate." in n or n.endswith(".router.weight")]
    for n, prm in model.named_parameters():
        if n in frozen:
            prm.requires_grad_(False)
    trainable = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
    print(f"[NeutronAI] plain transformers full fine-tune: {trainable / 1e9:.2f}B trainable weights; "
          f"{len(frozen)} router weights frozen", flush=True)
    return model, tokenizer


def make_full_ft_optimizer(model, lr: float, weight_decay: float = 0.01):
    """8-bit AdamW with bf16 stochastic rounding (torchao). The weights stay in bf16 (no fp32 copy: it wouldn't fit),
    and in plain bf16 an update smaller than about 1/256 of a weight rounds away to nothing; at full fine-tune
    learning rates most updates are that small. Stochastic rounding keeps them on average. 8-bit moments: ~2 bytes
    per weight. Weight decay on matrices only."""
    from torchao.optim import AdamW8bit

    named = [(n, prm) for n, prm in model.named_parameters() if prm.requires_grad]
    groups = [{"params": [prm for _, prm in named if prm.dim() >= 2], "weight_decay": weight_decay},
              {"params": [prm for _, prm in named if prm.dim() < 2], "weight_decay": 0.0}]
    print(f"[NeutronAI] optimizer: torchao AdamW8bit, bf16 stochastic rounding, lr {lr}", flush=True)
    return AdamW8bit(groups, lr=lr, bf16_stochastic_round=True)


def load_plain_lora(model_name: str, rank: int = 32):
    """bf16 model + LoRA with plain transformers + PEFT, for architectures Unsloth doesn't cover. Mixture-of-experts
    models whose experts are fused 3-D tensors (JetBrains Mellum2: mlp.experts.gate_up_proj / down_proj) get LoRA on
    those tensors through PEFT's `target_parameters`, plus the usual attention projections."""
    from peft import LoraConfig, get_peft_model

    model, tokenizer = _load_plain_bf16(model_name)
    model.enable_input_require_grads()
    experts = sorted({".".join(n.split(".")[-3:]) for n, prm in model.named_parameters() if prm.dim() == 3})
    names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    attn = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj") if m in names]
    dense_mlp = [] if experts else [m for m in ("gate_proj", "up_proj", "down_proj") if m in names]
    cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=attn + dense_mlp, **({"target_parameters": experts} if experts else {}))
    model = get_peft_model(model, cfg)
    trainable = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
    print(f"[NeutronAI] plain transformers + PEFT LoRA r={rank}: modules {attn + dense_mlp}, expert tensors {experts}; "
          f"{trainable / 1e6:.0f}M trainable weights", flush=True)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "fim-sft", "dpo", "grpo", "fim-grpo", "repo-grpo"], default="grpo")
    ap.add_argument("--holes-per-step", type=int, default=4, help="repo-grpo: holes per optimizer step")
    ap.add_argument("--lr", type=float, default=0.0, help="Learning rate override (fim-sft default 1e-4)")
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
    ap.add_argument("--fim-format", choices=["auto", "raw", "chat"], default="auto",
                    help="fim-grpo prompts: raw <|fim_prefix|> string (base models) or chat-wrapped; auto = raw unless Instruct")
    ap.add_argument("--export-gguf", action="store_true")
    ap.add_argument("--push-hf", default="")
    ap.add_argument("--out", default="hadron-v0.9")
    ap.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                    help="Qwen3/3.5 thinking mode. auto = on only if the completion budget (max-seq-len - 1024) >= 8192")
    ap.add_argument("--load-in-4bit", choices=["auto", "yes", "no"], default="auto",
                    help="auto = 4-bit below 40GB VRAM, bf16 on bigger GPUs (A100 80GB)")
    ap.add_argument("--full-finetune", action="store_true",
                    help="Train every weight (bf16, 8-bit AdamW) instead of a LoRA adapter. fim-sft only; for small "
                         "models (Quark 1.5B needs ~20GB VRAM + activations)")
    ap.add_argument("--batch-size", type=int, default=4, help="fim-sft: rows per GPU micro-batch")
    ap.add_argument("--plain-hf", action="store_true",
                    help="fim-sft/sft LoRA with plain transformers + PEFT instead of Unsloth (automatic for "
                         "architectures in PLAIN_HF_TYPES, e.g. JetBrains Mellum2)")
    ap.add_argument("--grad-accum", type=int, default=4, help="fim-sft: micro-batches per optimizer step")
    ap.add_argument("--fim-style", choices=["auto", "qwen-spm"], default="auto",
                    help="fim-sft prompt format: auto = the model's own (from its tokenizer); qwen-spm = Qwen's FIM "
                         "tokens with the suffix first (the pushed repo gets a quark_fim.json saying so)")
    args = ap.parse_args()
    if args.full_finetune and args.mode != "fim-sft":
        sys.exit("[NeutronAI] --full-finetune is only wired up for --mode fim-sft.")
    if args.push_hf:
        hf_preflight(args.push_hf)

    # ponytail: single-GPU Unsloth handles 1.5B-32B; upgrade to multi-node verl/OpenRLHF when funded past 1x H100.
    import torch
    plain = args.plain_hf or hf_model_type(args.model) in PLAIN_HF_TYPES
    resume_from = _find_resume_checkpoint(args.out, args.push_hf) if args.resume else None
    if plain:
        if args.mode not in ("fim-sft", "sft"):
            sys.exit("[NeutronAI] the plain transformers path is wired up for fim-sft and sft only.")
        model, tokenizer = (load_plain_full(args.model, resume_from) if args.full_finetune
                            else load_plain_lora(args.model))
        is_bfloat16_supported = torch.cuda.is_bf16_supported
        vision = False
    else:
        from unsloth import FastLanguageModel, FastModel, PatchDPOTrainer, is_bfloat16_supported

        vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
        # Use vLLM fast_inference on >=24GB single GPUs; on 16GB cards (e.g. A16/T4) use Unsloth native generation to avoid OOM!
        use_vllm = (args.mode in ("grpo", "fim-grpo", "repo-grpo")) and (vram_gb >= 22.0)
        if vram_gb > 0 and vram_gb < 22.0 and args.num_generations > 4:
            args.num_generations = 4
        load_4bit = not args.full_finetune and (args.load_in_4bit == "yes" or (args.load_in_4bit == "auto" and vram_gb < 40.0))
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
        if args.full_finetune:
            load_kwargs.update(full_finetuning=True, load_in_4bit=False)
        model, tokenizer = loader.from_pretrained(**load_kwargs)
        if args.full_finetune:
            print(f"[NeutronAI] full fine-tune: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B "
                  f"trainable weights", flush=True)
        elif not hasattr(model, "peft_config"):
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
    callbacks.append(_make_memory_report())
    if args.push_hf:
        callbacks.append(_make_hub_checkpoint_callback(args.push_hf))
    # Full fine-tune checkpoints hold every weight + optimizer state (~10 GB for 1.5B, ~50 GB for Quark-12B), so
    # upload them less often.
    save_steps = args.save_steps or (min(500, max(50, args.steps // 10)) if args.full_finetune
                                     else min(50, max(10, args.steps // 4)))

    if args.mode == "fim-sft":  # supervised fill-in-the-middle: learn the original code for each hole
        from transformers import Trainer, TrainingArguments

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        lr = args.lr or (2e-5 if args.full_finetune else 1e-4)  # every weight moves: smaller steps
        trainer = Trainer(
            model=model,
            train_dataset=load_fim_sft_dataset(args.data, tokenizer, args.max_seq_len, args.fim_style),
            data_collator=_fim_sft_collator(pad_id),
            callbacks=callbacks,
            # plain full fine-tune: our own optimizer (bf16 stochastic rounding); the Trainer still makes the schedule
            optimizers=(make_full_ft_optimizer(model, lr) if plain and args.full_finetune else None, None),
            args=TrainingArguments(
                output_dir=args.out,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=lr,
                optim="adamw_8bit" if args.full_finetune and not plain else "adamw_torch",  # unused if ours is given
                weight_decay=0.01 if args.full_finetune else 0.0,
                train_sampling_strategy="group_by_length",  # similar-length rows per batch: far less padding
                lr_scheduler_type="cosine",
                warmup_steps=max(1, args.steps // 20),  # 5% (transformers 5 dropped warmup_ratio)
                max_steps=args.steps,
                logging_steps=5,
                save_steps=save_steps,
                # a full fine-tune checkpoint is ~50 GB (weights + optimizer state): keep only the newest
                save_total_limit=1 if args.full_finetune else 2,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                remove_unused_columns=False,
                report_to="none",
            ),
        )
    elif args.mode == "sft":
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=load_sft_dataset(args.data, tokenizer),
            callbacks=callbacks,
            args=SFTConfig(
                output_dir=args.out,
                dataset_text_field="text",
                max_length=args.max_seq_len,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                warmup_steps=max(1, args.steps // 10),  # 10% (transformers 5 dropped warmup_ratio)
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
            learning_rate=args.lr or 5e-6,
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

        if args.mode == "repo-grpo":  # real-repo holes, rewarded by the repo's own tests (repo_fim.py, RepoChecker)
            from datasets import Dataset
            from repo_fim import repo_grpo_reward_func

            # Repo prompts are ~1,000 tokens; a prompt cut from the left would lose its <|fim_prefix|> token, so give
            # prompts their own budget and drop the few that don't fit. Several holes per step: their tests run at once.
            cfg_kwargs.update(max_prompt_length=args.max_prompt_tokens,
                              max_completion_length=args.max_seq_len - args.max_prompt_tokens,
                              gradient_accumulation_steps=args.num_generations * args.holes_per_step)
            with open(args.data, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]  # already in repo blocks: keep the file order
            before = len(rows)
            rows = [r for r in rows if len(tokenizer(r["fim_prompt"])["input_ids"]) <= args.max_prompt_tokens]
            print(f"[NeutronAI] repo-GRPO holes: {len(rows)} (dropped {before - len(rows)} with long prompts), "
                  f"{args.holes_per_step} per step", flush=True)
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[repo_grpo_reward_func],
                callbacks=callbacks,
                args=GRPOConfig(**cfg_kwargs),
                train_dataset=Dataset.from_list([
                    {"prompt": r["fim_prompt"], "repo": r["repo"], "path": r["path"], "lo": r["lo"], "hi": r["hi"],
                     "oracle_middle": r["oracle_middle"], "next_repo": r.get("next_repo", "")}
                    for r in rows
                ]),
            )
        elif args.mode == "fim-grpo":
            from datasets import Dataset
            from fim import fim_grpo_reward_func

            with open(args.data, encoding="utf-8") as f:
                fim_rows = [json.loads(l) for l in f if l.strip()]
            if args.max_prompt_tokens:  # ~3.5 chars per code token; skip whole files too long for the prompt budget
                before = len(fim_rows)
                fim_rows = [r for r in fim_rows if len(r.get("fim_prompt", "")) <= args.max_prompt_tokens * 3.5]
                print(f"[NeutronAI] FIM tasks: {len(fim_rows)} (dropped {before - len(fim_rows)} with long prompts)", flush=True)
            # No measured difficulty for FIM tasks, and the file is grouped by language: mix it (fixed seed, so
            # --resume sees the same order) since shuffle_dataset is off to keep Hadron's curriculum order.
            random.Random(42).shuffle(fim_rows)
            if any("baseline_pass_rate" in r for r in fim_rows):  # filtered by `fim.py filter`: ~50% holes first
                fim_rows.sort(key=lambda r: abs(r.get("baseline_pass_rate", 0.5) - 0.5))
                print("[NeutronAI] FIM curriculum: holes the model solves about half the time first", flush=True)
            fmt = render or (lambda messages: messages)
            # Base FIM models (Qwen2.5-Coder-1.5B etc.) are trained on the raw `<|fim_prefix|>...<|fim_middle|>` string,
            # exactly what an editor sends at the cursor; chat-wrapping it would train a format nobody will use.
            raw_fim = args.fim_format == "raw" or (args.fim_format == "auto" and "instruct" not in args.model.lower())
            print(f"[NeutronAI] FIM prompt format: {'raw <|fim_prefix|>' if raw_fim else 'chat'}", flush=True)
            fim_ds = Dataset.from_list([
                {
                    "prompt": r["fim_prompt"] if raw_fim else fmt([
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

    trainer.train(resume_from_checkpoint=resume_from)
    if torch.cuda.is_available():
        print(f"[NeutronAI] peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    for cb in callbacks:  # a checkpoint upload still running must finish before this process exits
        for fut in getattr(cb, "pending", []):
            if not fut.done():
                print("[NeutronAI] waiting for the checkpoint upload to finish...", flush=True)
            fut.result()
    if trainer.state.global_step < args.steps:  # stopped by --max-minutes: the run isn't finished
        print(f"[NeutronAI] stopped at step {trainer.state.global_step} of {args.steps}; the checkpoint is saved"
              + (f" on {args.push_hf}/last-checkpoint" if args.push_hf else "")
              + ". Run again with --resume to continue.", flush=True)
        return
    if not (plain and args.full_finetune):  # a plain full fine-tune is uploaded straight from memory
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)

    if args.push_hf and args.fim_style != "auto":  # tell fim.fim_style (benchmarks, editors) the prompt format
        from huggingface_hub import HfApi

        from fim import FIM_STYLE_FILE

        HfApi().create_repo(args.push_hf, exist_ok=True)
        HfApi().upload_file(path_or_fileobj=json.dumps({"style": args.fim_style}).encode(),
                            path_in_repo=FIM_STYLE_FILE, repo_id=args.push_hf,
                            commit_message=f"FIM prompt format: {args.fim_style}")
    if args.push_hf and plain and args.full_finetune:
        model.config.use_cache = True  # off only for training
        model.push_to_hub(args.push_hf, commit_message="Full fine-tune result")
        tokenizer.push_to_hub(args.push_hf)
        print(f"[NeutronAI] full fine-tune uploaded -> {args.push_hf}", flush=True)
        # The ~50 GB training checkpoint isn't part of the model: drop it so downloads (vLLM, llama.cpp) skip it
        from huggingface_hub import HfApi

        try:
            HfApi().delete_folder("last-checkpoint", repo_id=args.push_hf,
                                  commit_message="remove the training checkpoint")
        except Exception as e:
            print(f"[NeutronAI] (no training checkpoint to remove: {e})", flush=True)
    elif args.push_hf and plain:
        try:  # fold the LoRA into the weights and upload a normal model (vLLM / llama.cpp can load it directly)
            merged = model.merge_and_unload()
            merged.push_to_hub(args.push_hf, commit_message="LoRA fine-tune, merged")
            tokenizer.push_to_hub(args.push_hf)
        except Exception as e:  # never lose a finished run: at least upload the LoRA adapter
            print(f"[NeutronAI] Merged upload failed ({e}); uploading the LoRA adapter instead.", flush=True)
            from huggingface_hub import HfApi

            HfApi().upload_folder(repo_id=args.push_hf, folder_path=args.out, path_in_repo="lora_adapter")
        if args.export_gguf:
            print("[NeutronAI] GGUF export skipped on the plain transformers path (Unsloth's exporter); convert the "
                  "merged model with llama.cpp's convert_hf_to_gguf.py instead.", flush=True)
    elif args.push_hf and args.full_finetune:  # no adapter to merge: the model itself is the result
        model.push_to_hub(args.push_hf, commit_message="Full fine-tune result")
        tokenizer.push_to_hub(args.push_hf)
        if args.export_gguf:
            try:
                model.push_to_hub_gguf(f"{args.push_hf}-GGUF", tokenizer, quantization_method="q4_k_m")
            except Exception as e:
                print(f"[NeutronAI] GGUF export/upload failed ({e}); the bf16 model is already uploaded.", flush=True)
    elif args.push_hf:
        try:
            model.push_to_hub_merged(args.push_hf, tokenizer, save_method="merged_16bit")
            if vision:  # vLLM needs the base model's processor files even for text-only use
                try:
                    from hf_processor import copy_processor_files

                    copy_processor_files(args.model, args.push_hf)
                except Exception as e:
                    print(f"[NeutronAI] Could not copy processor files ({e}); fix with "
                          f"`uv run --with huggingface_hub hf_processor.py {args.model} {args.push_hf}`", flush=True)
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
