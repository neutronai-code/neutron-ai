"""Check that an optional fast kernel really works on this GPU (used by vultr.py before training).

  uv run python kernel_check.py conv   # causal-conv1d (Qwen3.5 / Gated DeltaNet short convolution)
  uv run python kernel_check.py fla    # flash-linear-attention (gated delta rule)

Exits 0 only if the kernel imports AND runs. A half-built kernel that imports but crashes would otherwise take
the whole training run down, so the VM uninstalls anything that fails this check.
"""
import sys

import torch


def check_conv():
    from causal_conv1d import causal_conv1d_update

    dim, width = 64, 4
    x = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(2, dim, width, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    out = causal_conv1d_update(x, state, w, None, "silu")
    torch.cuda.synchronize()
    assert out.shape == x.shape and torch.isfinite(out.float()).all()


def check_fla():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    b, t, h, d = 1, 64, 2, 32
    q, k, v = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    g = -torch.rand(b, t, h, device="cuda", dtype=torch.float32)
    beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    torch.cuda.synchronize()
    assert o.shape == v.shape and torch.isfinite(o.float()).all()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        {"conv": check_conv, "fla": check_fla}[which]()
    except Exception as e:  # noqa: BLE001 - any failure means "don't use this kernel"
        print(f"[NeutronAI] fast kernel '{which}' not usable: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(f"[NeutronAI] fast kernel '{which}' OK", flush=True)
