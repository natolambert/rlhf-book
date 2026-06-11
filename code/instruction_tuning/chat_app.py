"""Minimal Gradio chat UI to play with the instruction-tuned checkpoints.

Run from code/:

    uv run --extra ui python -m instruction_tuning.chat_app

    # point at a different runs dir, or open to your LAN:
    RLHF_RUNS=/home/ghoti/rlhf-runs uv run --extra ui python -m instruction_tuning.chat_app --host 0.0.0.0 --port 7860

Each checkpoint saved by ``train_qwen_base.py`` is self-describing: it carries
its own chat template and ``generation_config`` stop ids, so this app loads
nothing model-specific. Switch checkpoints in the dropdown and feel the
difference — e.g. the v1 run (untrained ``<|im_end|>``, never stops) versus the
v2 warm-start run (clean stop) versus the raw base model (continues forever).

Two toggles tie the UI back to the training lessons:
  * "Greedy" turns sampling off (deterministic argmax decoding).
  * "Show special tokens" decodes with the role markers / EOS visible, so you
    can literally watch whether and where the model emits its stop token.
"""

import argparse
import os
import threading
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


RUNS_DIR = Path(os.environ.get("RLHF_RUNS", Path.home() / "rlhf-runs"))

# Reference models pulled from the Hub to compare against your checkpoints.
REFERENCE_MODELS = ["Qwen/Qwen3-1.7B-Base"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


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

# Lazily load one model at a time; switching frees the previous one's VRAM.
_cache: dict = {"name": None, "model": None, "tok": None}


def load(name: str):
    if _cache["name"] == name and _cache["model"] is not None:
        return _cache["model"], _cache["tok"]
    if _cache["model"] is not None:
        _cache["model"] = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    path = MODELS[name]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()
    _cache.update(name=name, model=model, tok=tok)
    return model, tok


def respond(message, history, model_name, temperature, top_p, max_new_tokens, greedy, show_special):
    """Stream a reply. history is a list of {'role','content'} (messages format)."""
    if not model_name:
        yield "Pick a model in the dropdown first."
        return
    model, tok = load(model_name)

    msgs = [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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

    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    out = ""
    for chunk in streamer:
        out += chunk
        yield out


def build_demo() -> gr.Blocks:
    choices = list(MODELS)
    # Default to the v2 warm-start checkpoint if present, else the first entry.
    default = next(
        (c for c in choices if "v2" in c or "warmstart" in c), choices[0] if choices else None
    )

    with gr.Blocks(title="RLHF Book — SFT playground") as demo:
        gr.Markdown(
            "# RLHF Book — instruction-tuning playground\n"
            "Chat with your SFT checkpoints. Switch models to compare base vs v1 vs v2. "
            "Turn on **Show special tokens** to watch where the model emits its stop token."
        )
        model_dd = gr.Dropdown(choices=choices, value=default, label="Model")
        with gr.Accordion("Decoding settings", open=False):
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_new = gr.Slider(16, 1024, value=256, step=16, label="Max new tokens")
            greedy = gr.Checkbox(value=False, label="Greedy (deterministic, ignores temp/top-p)")
            show_special = gr.Checkbox(
                value=False, label="Show special tokens (reveal <|im_end|> etc.)"
            )

        # Gradio 6.x ChatInterface uses the messages format (list of {role, content})
        # by default and dropped the `type` argument; respond() already expects that.
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[model_dd, temperature, top_p, max_new, greedy, show_special],
            examples=[
                ["What is the capital of France?"],
                ["Explain quantum computing in simple terms."],
                ["Write a haiku about programming."],
            ],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Chat UI for SFT checkpoints.")
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
