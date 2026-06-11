"""Gradio UI to play with and inspect the instruction-tuned checkpoints.

Run from code/:

    uv run --extra ui python -m instruction_tuning.chat_app

    # point at a different runs dir, or open to your LAN:
    RLHF_RUNS=/home/ghoti/rlhf-runs uv run --extra ui python -m instruction_tuning.chat_app --host 0.0.0.0 --port 7860

Each checkpoint saved by ``train_qwen_base.py`` is self-describing: it carries
its own chat template and ``generation_config`` stop ids.

Three tabs:
  * Inspect — type a sentence + pick a model and see EVERY stage: the exact
    string the model receives after its pipeline, the tokenized input with
    special tokens highlighted, the streamed output, and why it stopped. The
    pipeline is auto-chosen per model (raw completion for a base model, chat
    template for an SFT checkpoint) and can be overridden.
  * Chat — normal multi-turn chat with one checkpoint.
  * Compare — the same prompt through two models side by side.
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

# Reference (base) models pulled from the Hub to compare against your checkpoints.
# Anything listed here defaults to the raw-completion pipeline under "Auto".
REFERENCE_MODELS = ["Qwen/Qwen3-1.7B-Base"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Compare needs two models resident at once; a 1.7B bf16 model is ~3.5 GB, so a
# few fit easily on a 32 GB card. Cache them and evict least-recently-used.
MAX_LOADED = 3

PIPELINE_AUTO = "Auto (base → raw, SFT → chat)"
PIPELINE_CHAT = "Chat (apply template)"
PIPELINE_RAW = "Raw (completion, no template)"


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
        models[f"[base] {m}"] = m
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


def resolve_pipeline(choice: str, model_name: str) -> str:
    """Return 'chat' or 'raw' for a pipeline choice (Auto keys off model kind)."""
    if choice == PIPELINE_CHAT:
        return "chat"
    if choice == PIPELINE_RAW:
        return "raw"
    # Auto: base reference models -> raw completion, SFT checkpoints -> chat.
    return "raw" if model_name.startswith("[base]") else "chat"


def build_input(sentence: str, tok, pipeline: str):
    """Return (display_string, input_ids[1, T]) for the chosen pipeline.

    The display string is exactly what gets tokenized, so the "model input"
    panel and the token view always agree. For chat we render the template to
    text first, then tokenize with add_special_tokens=False (the template text
    already contains the special markers; this is the same path generate uses).
    """
    if pipeline == "raw":
        display = sentence
        input_ids = tok(sentence, return_tensors="pt").input_ids
    else:
        display = tok.apply_chat_template(
            [{"role": "user", "content": sentence}], tokenize=False, add_generation_prompt=True
        )
        input_ids = tok(display, return_tensors="pt", add_special_tokens=False).input_ids
    return display, input_ids


def token_highlight(tok, ids):
    """List of (piece, 'special'|None) for gr.HighlightedText."""
    specials = set(tok.all_special_ids)
    return [(tok.decode([t]), "special" if t in specials else None) for t in ids]


def _stream(model, tok, input_ids, greedy, temperature, top_p, max_new_tokens, show_special):
    """Yield cumulative output text; stash the full output tensor in `box`."""
    box = {}
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=not show_special)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": not greedy,
        "streamer": streamer,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    if not greedy:
        kwargs.update(temperature=float(temperature), top_p=float(top_p))

    def run():
        with torch.no_grad():
            box["out"] = model.generate(**kwargs)

    thread = threading.Thread(target=run)
    thread.start()
    out = ""
    for chunk in streamer:
        out += chunk
        yield out, box
    thread.join()
    yield out, box


def run_inspect(
    sentence, model_name, pipeline_choice, greedy, temperature, top_p, max_new_tokens, show_special
):
    """Inspect tab: stream output plus full input/token/stop-reason transparency."""
    if not (model_name and sentence):
        yield "", [], "", "_pick a model and type a sentence_"
        return
    model, tok = load(model_name)
    pipeline = resolve_pipeline(pipeline_choice, model_name)
    display, input_ids = build_input(sentence, tok, pipeline)
    input_ids = input_ids.to(model.device)
    highlight = token_highlight(tok, input_ids[0].tolist())
    n_in = input_ids.shape[1]
    header = f"**pipeline:** `{pipeline}`  ·  **{n_in}** input tokens  ·  generating…"
    # First frame: input + tokens visible, output empty.
    yield display, highlight, "", header

    box = {}
    for out, box in _stream(
        model, tok, input_ids, greedy, temperature, top_p, max_new_tokens, show_special
    ):
        yield gr.update(), gr.update(), out, header

    # Stop reason.
    gen_ids = box["out"][0][n_in:].tolist()
    stop_ids = model.generation_config.eos_token_id
    stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
    if gen_ids and gen_ids[-1] in stop_ids:
        reason = f"stopped on EOS {gen_ids[-1]} (`{tok.decode([gen_ids[-1]])}`)"
    else:
        reason = f"hit max_new_tokens ({int(max_new_tokens)}) — no stop token emitted"
    final = (
        f"**pipeline:** `{pipeline}`  ·  **{n_in}** input + **{len(gen_ids)}** output tokens  ·  "
        f"{reason}  ·  stop ids `{stop_ids}`"
    )
    yield gr.update(), gr.update(), out, final


def respond(message, history, model_name, temperature, top_p, max_new_tokens, greedy, show_special):
    """Chat tab: build messages from history and stream one model's reply."""
    if not model_name:
        yield "Pick a model in the dropdown first."
        return
    model, tok = load(model_name)
    msgs = [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    input_ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    for out, _ in _stream(
        model, tok, input_ids, greedy, temperature, top_p, max_new_tokens, show_special
    ):
        yield out


def compare(prompt, model_a, model_b, temperature, top_p, max_new_tokens, greedy, show_special):
    """Compare tab: same prompt through two models, streaming column A then B."""
    if not (model_a and model_b):
        yield "Pick both models.", ""
        return
    a_out = ""
    for col, name in ((0, model_a), (1, model_b)):
        model, tok = load(name)
        pipeline = resolve_pipeline(PIPELINE_AUTO, name)
        _, input_ids = build_input(prompt, tok, pipeline)
        input_ids = input_ids.to(model.device)
        for out, _ in _stream(
            model, tok, input_ids, greedy, temperature, top_p, max_new_tokens, show_special
        ):
            if col == 0:
                a_out = out
                yield a_out, ""
            else:
                yield a_out, out


def build_demo() -> gr.Blocks:
    choices = list(MODELS)
    v2 = next((c for c in choices if "v2" in c or "warmstart" in c), None)
    base = next((c for c in choices if c.startswith("[base]")), None)
    chat_default = v2 or (choices[0] if choices else None)

    with gr.Blocks(title="RLHF Book — SFT playground") as demo:
        gr.Markdown(
            "# RLHF Book — instruction-tuning playground\n"
            "Inspect exactly what each model receives and produces, chat with a checkpoint, "
            "or compare two side by side."
        )

        with gr.Tab("Inspect"):
            gr.Markdown(
                "Type a sentence and pick a model. **Auto** uses raw completion for the base "
                "model and the chat template for your SFT checkpoints — so you only choose the "
                "input and the model. The panels show every stage of what runs."
            )
            with gr.Row():
                insp_model = gr.Dropdown(choices=choices, value=base or chat_default, label="Model")
                insp_pipe = gr.Radio(
                    [PIPELINE_AUTO, PIPELINE_CHAT, PIPELINE_RAW],
                    value=PIPELINE_AUTO,
                    label="Pipeline",
                )
            insp_input = gr.Textbox(value="The day after Monday is", label="Your input", lines=2)
            with gr.Accordion("Decoding settings", open=False):
                i_greedy = gr.Checkbox(value=True, label="Greedy (deterministic)")
                i_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                i_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                i_max = gr.Slider(16, 1024, value=128, step=16, label="Max new tokens")
                i_special = gr.Checkbox(value=True, label="Show special tokens in output")
            insp_btn = gr.Button("Run", variant="primary")
            insp_header = gr.Markdown()
            insp_modelinput = gr.Textbox(
                label="1 · What the model receives (after the pipeline)", lines=4
            )
            insp_tokens = gr.HighlightedText(
                label="2 · Tokenized input (special tokens highlighted)", combine_adjacent=False
            )
            insp_output = gr.Textbox(label="3 · Model output", lines=8)
            insp_btn.click(
                run_inspect,
                inputs=[
                    insp_input,
                    insp_model,
                    insp_pipe,
                    i_greedy,
                    i_temp,
                    i_top_p,
                    i_max,
                    i_special,
                ],
                outputs=[insp_modelinput, insp_tokens, insp_output, insp_header],
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
                "Same prompt, two models, each with its Auto pipeline. Defaults to **base vs v2** "
                "— the clearest before/after for the stop-token fix."
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
    parser = argparse.ArgumentParser(description="Inspect / chat / compare UI for SFT checkpoints.")
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
