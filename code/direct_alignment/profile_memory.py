# Memory profiling script for direct alignment training
#
# Tests different batch sizes and context lengths to find safe configurations
# that stay below 85% GPU memory usage.
#
# Usage:
#   uv run python -m direct_alignment.profile_memory

import gc
import platform
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_memory_usage() -> tuple[float, float]:
    """Get current memory usage in GB.

    Returns:
        (used_gb, total_gb)
    """
    # For unified memory systems (DGX Spark), use system memory
    result = subprocess.run(
        ["free", "-b"], capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    mem_line = lines[1].split()
    total_bytes = int(mem_line[1])
    used_bytes = int(mem_line[2])

    total_gb = total_bytes / (1024**3)
    used_gb = used_bytes / (1024**3)

    return used_gb, total_gb


def get_attn_implementation() -> str:
    """Determine the best attention implementation for this platform."""
    if platform.machine() != "x86_64":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def profile_configuration(
    model_name: str,
    batch_size: int,
    max_length: int,
    num_forward_passes: int = 3,
) -> dict:
    """Profile memory usage for a given configuration.

    Args:
        model_name: HuggingFace model name
        batch_size: Batch size to test
        max_length: Sequence length to test
        num_forward_passes: Number of forward passes to warm up

    Returns:
        Dictionary with memory metrics
    """
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    baseline_used, total_gb = get_memory_usage()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()

    # Load policy model (with gradient checkpointing)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy_model.train()

    # Load reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    models_loaded_used, _ = get_memory_usage()
    models_memory = models_loaded_used - baseline_used

    # Create dummy batch (DPO needs chosen + rejected)
    dummy_input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (batch_size, max_length),
        device="cuda"
    )
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-7)

    # Warm up with forward/backward passes
    peak_memory = 0
    for i in range(num_forward_passes):
        optimizer.zero_grad()

        # Policy forward (chosen)
        policy_outputs = policy_model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            use_cache=False,
        )
        policy_chosen_logits = policy_outputs.logits

        # Policy forward (rejected) - in real DPO, different sequences
        policy_outputs_rej = policy_model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            use_cache=False,
        )
        policy_rejected_logits = policy_outputs_rej.logits

        # Reference forward (chosen)
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                use_cache=False,
            )
            ref_chosen_logits = ref_outputs.logits

            ref_outputs_rej = ref_model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                use_cache=False,
            )
            ref_rejected_logits = ref_outputs_rej.logits

        # Simplified DPO loss (just to trigger backward)
        loss = (policy_chosen_logits.mean() - policy_rejected_logits.mean()).abs()
        loss.backward()
        optimizer.step()

        current_used, _ = get_memory_usage()
        peak_memory = max(peak_memory, current_used)

    # Final measurements
    final_used, _ = get_memory_usage()

    # Cleanup
    del policy_model, ref_model, optimizer, dummy_input_ids, dummy_attention_mask
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "batch_size": batch_size,
        "max_length": max_length,
        "baseline_gb": baseline_used,
        "models_memory_gb": models_memory,
        "peak_memory_gb": peak_memory,
        "total_memory_gb": total_gb,
        "peak_usage_pct": (peak_memory / total_gb) * 100,
    }


def main():
    model_name = "allenai/OLMo-2-0425-1B-SFT"

    # Configurations to test - focus on larger context lengths
    batch_sizes = [1, 2, 4, 8, 16]
    max_lengths = [512, 1024, 2048, 4096]

    print(f"Memory Profiling for {model_name}")
    print("=" * 80)

    _, total_gb = get_memory_usage()
    target_max_gb = total_gb * 0.85
    print(f"Total memory: {total_gb:.1f} GB")
    print(f"Target max (85%): {target_max_gb:.1f} GB")
    print()

    results = []

    for max_length in max_lengths:
        print(f"\n--- Context Length: {max_length} ---")
        for batch_size in batch_sizes:
            try:
                print(f"Testing batch_size={batch_size}, max_length={max_length}...", end=" ", flush=True)
                result = profile_configuration(
                    model_name=model_name,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                results.append(result)

                status = "OK" if result["peak_usage_pct"] < 85 else "OVER 85%"
                print(f"Peak: {result['peak_memory_gb']:.1f} GB ({result['peak_usage_pct']:.1f}%) [{status}]")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM!")
                    results.append({
                        "batch_size": batch_size,
                        "max_length": max_length,
                        "peak_memory_gb": float("inf"),
                        "peak_usage_pct": 100,
                        "error": "OOM",
                    })
                else:
                    raise

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Safe configurations (< 85% memory)")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Max Length':<12} {'Peak Memory':<15} {'Usage %':<10} {'Status':<10}")
    print("-" * 60)

    for r in results:
        if "error" in r:
            status = "OOM"
            peak = "N/A"
            pct = "N/A"
        else:
            status = "SAFE" if r["peak_usage_pct"] < 85 else "OVER"
            peak = f"{r['peak_memory_gb']:.1f} GB"
            pct = f"{r['peak_usage_pct']:.1f}%"

        print(f"{r['batch_size']:<12} {r['max_length']:<12} {peak:<15} {pct:<10} {status:<10}")

    # Recommended configuration
    safe_configs = [r for r in results if "error" not in r and r["peak_usage_pct"] < 85]
    if safe_configs:
        # Pick highest throughput (batch_size * max_length) that's safe
        best = max(safe_configs, key=lambda x: x["batch_size"] * x["max_length"])
        print(f"\nRecommended: batch_size={best['batch_size']}, max_length={best['max_length']}")
        print(f"  Peak memory: {best['peak_memory_gb']:.1f} GB ({best['peak_usage_pct']:.1f}%)")


if __name__ == "__main__":
    main()
