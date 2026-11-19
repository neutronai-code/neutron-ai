import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Why did Quark-12B v0.6 get worse? Measures, with the exact training code path (transformers + PEFT, sdpa,
    # grouped_mm experts), the loss on the original code of 100 training rows and 100 benchmark holes for:
    # untrained Mellum2, the adapter after 190 steps and after all 3714 steps, in eval and train mode; plus the
    # untrained model with eager attention / eager experts, to catch a training-vs-inference mismatch.
    # Turn the GPU on (restart the notebook first if another model is loaded), paste the HF token, click Start.
    import marimo as mo

    token = mo.ui.text(kind="password", label="Hugging Face token (hf_...)", full_width=True)
    start = mo.ui.run_button(label="Start diagnosis")
    mo.vstack([token, start])
    return mo, start, token


@app.cell
def _(mo, start, token):
    import gc, json, os, random, subprocess, sys, time

    mo.stop(not start.value, mo.md("Paste your Hugging Face token above, then click **Start diagnosis**."))
    tok = token.value.strip()
    assert tok.startswith("hf_"), "Paste your Hugging Face token first"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "peft==0.21.3",
                    "accelerate"], check=True)
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    BASE = "JetBrains/Mellum2-12B-A2.5B-Base"
    TUNED = "neutronai-code/Quark-12B-A2.5B-v0.6"
    work = os.path.abspath("quark_work")
    os.makedirs(work, exist_ok=True)
    results = []

    def note(label, value):
        results.append((label, value))
        print(f"{label:62} {value}", flush=True)

    # Qwen-format FIM prompt -> Mellum's (same as fim.to_mellum_prompt, inlined so no code download is needed)
    def to_mellum(p):
        i = p.index("<|fim_prefix|>")
        ctx, rest = p[:i], p[i + len("<|fim_prefix|>"):]
        pre, rest = rest.split("<|fim_suffix|>", 1)
        suf = rest[:rest.rindex("<|fim_middle|>")]
        return ctx.replace("<|file_sep|>", "<filename>") + "<fim_suffix>" + suf + "<fim_prefix>" + pre + "<fim_middle>"

    # 1. Data: 100 training rows (fill-only loss, like most rows) and 100 benchmark holes, tokenized like train.py
    tokz = AutoTokenizer.from_pretrained(BASE)
    eos = tokz.eos_token

    def enc(rows):
        out = []
        for r in rows:
            p = tokz(to_mellum(r["fim_prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
            c = tokz(r["oracle_middle"] + eos, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if p.shape[1] + c.shape[1] <= 4096:
                out.append((p, c, r))
        return out

    rng = random.Random(0)
    train_path = hf_hub_download("neutronai-code/quark-data", "fim_v05.jsonl", repo_type="dataset", token=tok, local_dir=work)
    with open(train_path, encoding="utf-8") as f:
        train_rows = [json.loads(l) for l in f if l.strip()]
    train_rows = [r for r in train_rows if r.get("loss") != "all"]
    train = enc(rng.sample(train_rows, 100))
    del train_rows
    bench_path = hf_hub_download("neutronai-code/quark-data", "bench/repo_holes_ctx2.jsonl", repo_type="dataset",
                                 token=tok, local_dir=work)
    with open(bench_path, encoding="utf-8") as f:
        bench_rows = [json.loads(l) for l in f if l.strip()]
    bench = enc(rng.sample(bench_rows, 100))
    long_frac = sum(p.shape[1] > 1024 for p, _, _ in bench) / len(bench)
    print(f"{len(train)} training rows, {len(bench)} benchmark holes; {long_frac:.0%} of benchmark prompts are longer "
          f"than Mellum2's 1024-token sliding window", flush=True)

    def loss_on(model, items, train_mode=False):
        model.train(train_mode)
        tot = cnt = 0.0
        with torch.no_grad():
            for p, c, _ in items:
                x = torch.cat([p, c], 1).to("cuda")
                labels = x.clone()
                labels[:, :p.shape[1]] = -100
                l = model(input_ids=x, labels=labels, use_cache=False).loss.item()
                tot += l * c.shape[1]
                cnt += c.shape[1]
        model.eval()
        return round(tot / cnt, 3)

    def load(attn="sdpa", experts="grouped_mm"):
        kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn)
        try:
            return AutoModelForCausalLM.from_pretrained(BASE, experts_implementation=experts, **kw).eval()
        except (TypeError, ValueError) as e:
            print(f"experts_implementation={experts} not accepted ({e}); default used", flush=True)
            return AutoModelForCausalLM.from_pretrained(BASE, **kw).eval()

    def sample(model, item, n=160):
        p, _, r = item
        x = p.to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                                 use_cache=True, pad_token_id=tokz.eos_token_id)
        return tokz.decode(out[0, x.shape[1]:], skip_special_tokens=False)[:300]

    # 2. Untrained Mellum2, exactly as training loads it (sdpa attention, grouped_mm experts)
    t0 = time.time()
    model = load()
    note("untrained, eval: training rows", loss_on(model, train))
    note("untrained, eval: benchmark holes", loss_on(model, bench))
    note("untrained, TRAIN mode: training rows", loss_on(model, train, train_mode=True))
    s_base = sample(model, bench[0])

    # 3. The trained adapters (PEFT, unmerged: the same forward as training), after 190 and 3714 steps
    commits = {c.title: c.commit_id for c in HfApi(token=tok).list_repo_commits(TUNED)}
    model = PeftModel.from_pretrained(model, TUNED, subfolder="last-checkpoint", adapter_name="final", token=tok)
    model.load_adapter(TUNED, subfolder="last-checkpoint", adapter_name="s190",
                       revision=commits["checkpoint step 190"], token=tok)
    for name, label in (("s190", "after 190 steps"), ("final", "after 3714 steps")):
        model.set_adapter(name)
        note(f"{label}, eval: training rows", loss_on(model, train))
        note(f"{label}, eval: benchmark holes", loss_on(model, bench))
        note(f"{label}, TRAIN mode: training rows", loss_on(model, train, train_mode=True))
    model.set_adapter("final")
    s_tuned = sample(model, bench[0])
    with model.disable_adapter():
        note("adapter disabled again (should equal untrained): bench", loss_on(model, bench))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Untrained Mellum2 with other attention / expert code on the 40 longest benchmark prompts: if these differ a
    # lot from sdpa + grouped_mm, training computed a different model than inference runs
    longest = sorted(bench, key=lambda it: -it[0].shape[1])[:40]
    for attn, experts in (("sdpa", "grouped_mm"), ("eager", "grouped_mm"), ("sdpa", "eager")):
        m = load(attn, experts)
        note(f"untrained, attention={attn}, experts={experts}: 40 longest", loss_on(m, longest))
        del m
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n--- benchmark hole 1, original:\n{bench[0][2]['oracle_middle'][:300]}\n--- untrained fill:\n{s_base}"
          f"\n--- trained fill:\n{s_tuned}\n\n(done in {time.time() - t0:.0f}s) Copy everything above to Claude.")
    mo.md("| measurement | loss |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in results))
    return


if __name__ == "__main__":
    app.run()
