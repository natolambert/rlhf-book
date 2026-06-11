"""Gradio UI to play with the instruction-tuned checkpoints.

Run from code/:

    uv run --extra ui python -m instruction_tuning.chat_app

    # point at a different runs dir, or open to your LAN:
    RLHF_RUNS=/home/ghoti/rlhf-runs uv run --extra ui python -m instruction_tuning.chat_app --host 0.0.0.0 --port 7860

Each checkpoint saved by ``train_qwen_base.py`` is self-describing: it carries
its own chat template and ``generation_config`` stop ids, so this app loads
nothing model-specific.

Two tabs:
  * Chat — a normal multi-turn chat with one checkpoint.
  * Compare — the same prompt through two models side by side (defaults to base
    vs the v2 warm-start checkpoint), so the base/v1/v2 difference is one click.

Two toggles tie the UI back to the training lessons:
  * "Greedy" turns sampling off (deterministic argmax decoding).
  * "Show special tokens" decodes with the role markers / EOS visible, so you
    can literally watch whether and where the model emits its stop token.
"""

import argparse
import os
import threading
from collections import OrderedDict
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


RUNS_DIR = Path(os.environ.get("RLHF_RUNS", Path.home() / "rlhf-runs"))

# Reference models pulled from the Hub to compare against your checkpoints.
REFERENCE_MODELS = ["Qwen/Qwen3-1.7B-Base"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Compare needs two models resident at once; a 1.7B bf16 model is ~3.5 GB, so a
# few fit easily on a 32 GB card. Cache them and evict least-recently-used.
MAX_LOADED = 3


def discover_models() -> dict[str, str]:
    """Map display name -> path/id for every loadable checkpoint + references."""
    models: dict[str, str] = {}
    if RUNS_DIR.is_dir():
        for d in sorted(RUNS_DIR.iterdir()):
            if not d.is_dir():
                continue
            has_weights = any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists()
            if has_weights and (d / "config.json").exists():
                models[d.name] = str(d)
    for m in REFERENCE_MODELS:
        models[f"[hub] {m}"] = m
    return models


MODELS = discover_models()

# LRU cache of loaded (model, tokenizer) pairs, keyed by display name.
_cache: "OrderedDict[str, tuple]" = OrderedDict()


def load(name: str):
    if name in _cache:
        _cache.move_to_end(name)
        return _cache[name]
    path = MODELS[name]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()
    _cache[name] = (model, tok)
    _cache.move_to_end(name)
    while len(_cache) > MAX_LOADED:
        _, (old_model, _) = _cache.popitem(last=False)
        del old_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return _cache[name]


def _gen_stream(messages, model_name, temperature, top_p, max_new_tokens, greedy, show_special):
    """Core streaming generator: yields the cumulative decoded reply string."""
    model, tok = load(model_name)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=not show_special)
    kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=not greedy,
        streamer=streamer,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    if not greedy:
        kwargs.update(temperature=float(temperature), top_p=float(top_p))
    threading.Thread(target=model.generate, kwargs=kwargs).start()
    out = ""
    for chunk in streamer:
        out += chunk
        yield out


def respond(message, history, model_name, temperature, top_p, max_new_tokens, greedy, show_special):
    """Chat tab: build messages from history and stream one model's reply."""
    if not model_name:
        yield "Pick a model in the dropdown first."
        return
    msgs = [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    yield from _gen_stream(
        msgs, model_name, temperature, top_p, max_new_tokens, greedy, show_special
    )


def compare(prompt, model_a, model_b, temperature, top_p, max_new_tokens, greedy, show_special):
    """Compare tab: same prompt through two models, streaming column A then B."""
    if not (model_a and model_b):
        yield "Pick both models.", ""
        return
    msgs = [{"role": "user", "content": prompt}]
    a_out = ""
    for a in _gen_stream(msgs, model_a, temperature, top_p, max_new_tokens, greedy, show_special):
        a_out = a
        yield a_out, ""
    for b in _gen_stream(msgs, model_b, temperature, top_p, max_new_tokens, greedy, show_special):
        yield a_out, b


def build_demo() -> gr.Blocks:
    choices = list(MODELS)
    v2 = next((c for c in choices if "v2" in c or "warmstart" in c), None)
    base = next((c for c in choices if c.startswith("[hub]")), None)
    chat_default = v2 or (choices[0] if choices else None)

    with gr.Blocks(title="RLHF Book — SFT playground") as demo:
        gr.Markdown(
            "# RLHF Book — instruction-tuning playground\n"
            "Chat with your SFT checkpoints, or compare two side by side. Turn on "
            "**Show special tokens** to watch where each model emits its stop token."
        )

        with gr.Tab("Chat"):
            model_dd = gr.Dropdown(choices=choices, value=chat_default, label="Model")
            with gr.Accordion("Decoding settings", open=False):
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                max_new = gr.Slider(16, 1024, value=256, step=16, label="Max new tokens")
                greedy = gr.Checkbox(value=False, label="Greedy (deterministic)")
                show_special = gr.Checkbox(value=False, label="Show special tokens")
            gr.ChatInterface(
                fn=respond,
                additional_inputs=[model_dd, temperature, top_p, max_new, greedy, show_special],
                examples=[
                    ["What is the capital of France?"],
                    ["Explain quantum computing in simple terms."],
                    ["Write a haiku about programming."],
                ],
            )

        with gr.Tab("Compare two models"):
            gr.Markdown(
                "Same prompt, two models. Defaults to **base vs v2** — the clearest "
                "before/after for the stop-token fix. Greedy + special tokens are on so "
                "the comparison is deterministic and you can see the `<|im_end|>` (or its "
                "absence)."
            )
            with gr.Row():
                cmp_a = gr.Dropdown(choices=choices, value=base or chat_default, label="Model A")
                cmp_b = gr.Dropdown(choices=choices, value=v2 or chat_default, label="Model B")
            cmp_prompt = gr.Textbox(value="What is the capital of France?", label="Prompt", lines=2)
            with gr.Accordion("Decoding settings", open=False):
                c_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                c_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                c_max = gr.Slider(16, 1024, value=256, step=16, label="Max new tokens")
                c_greedy = gr.Checkbox(value=True, label="Greedy (deterministic)")
                c_special = gr.Checkbox(value=True, label="Show special tokens")
            cmp_btn = gr.Button("Generate from both", variant="primary")
            with gr.Row():
                out_a = gr.Textbox(label="Model A output", lines=14)
                out_b = gr.Textbox(label="Model B output", lines=14)
            cmp_btn.click(
                compare,
                inputs=[cmp_prompt, cmp_a, cmp_b, c_temp, c_top_p, c_max, c_greedy, c_special],
                outputs=[out_a, out_b],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Chat / compare UI for SFT checkpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    if not MODELS:
        raise SystemExit(
            f"No checkpoints found under {RUNS_DIR} and no reference models configured. "
            "Train one first or set RLHF_RUNS."
        )
    print(f"Discovered {len(MODELS)} model(s): {', '.join(MODELS)}")
    build_demo().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
