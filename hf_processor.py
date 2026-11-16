"""Copy the image/video processor files from a base model repo into a trained model repo.

Unsloth's merged upload of a vision-architecture model (Qwen3.5) only writes the weights, config and tokenizer.
vLLM refuses to load it without `preprocessor_config.json` ("Can't load image processor"), even for text-only use.
train.py calls this after every upload; to repair an existing repo:

  uv run --with huggingface_hub hf_processor.py Qwen/Qwen3.5-4B neutronai-code/Hadron-4B-smoke
"""
import sys

PROCESSOR_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "LICENSE",  # Apache-2.0 asks redistributions to include it
)


def copy_processor_files(base_repo: str, target_repo: str) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    base_files = set(api.list_repo_files(base_repo))
    target_files = set(api.list_repo_files(target_repo))
    copied = []
    for name in PROCESSOR_FILES:
        if name in base_files and name not in target_files:
            local = hf_hub_download(base_repo, name)
            api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=target_repo,
                            commit_message=f"Add {name} from {base_repo}")
            copied.append(name)
    print(f"[NeutronAI] processor files copied {base_repo} -> {target_repo}: {copied or 'none needed'}", flush=True)
    return copied


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: uv run --with huggingface_hub hf_processor.py <base_repo> <target_repo>")
    copy_processor_files(sys.argv[1], sys.argv[2])
